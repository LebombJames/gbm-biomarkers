from __future__ import annotations
import ants
from ants import ANTsImage
import tiffslide
import numpy as np
from PIL import Image
import SimpleITK as sitk

import gc
from typing import overload, Literal, Any
from typing_extensions import TypeIs
from pathlib import Path
from functools import cached_property
import subprocess
import os

from __types import *

import ants.config

ants.config.set_ants_deterministic(True, 123)


class LazyAntsImage:
    """
    A wrapper around Path that lazy loads an AntsImage when called, or returns a cached version.
    """

    def __init__(
        self,
        path_or_img: Path | ANTsImage,
        level: int = 2,
        *args,
        **kwargs,
    ) -> None:
        self.level = level
        self.args = args
        self.kwargs = kwargs

        self.maps = {}

        if isinstance(path_or_img, Path):
            self.path = path_or_img
        elif isinstance(path_or_img, ANTsImage):
            self.img = path_or_img
        else:
            raise ValueError("Neither a Path or an AntsImage were provided.")

    @cached_property
    def img(self) -> ANTsImage:
        """Return the cached Ants image, or load it from the path if not loaded yet."""

        if self.path is None:
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately before
            # this whole function is even called.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        print(f"Loading image: {self.path}")

        return (
            self.svs_read(self.path)
            if self.is_hist
            else ants.image_read(str(self.path), *self.args, **self.kwargs)
        )

    @property
    def is_hist(self):
        return str(self.path).endswith(".svs")

    @property
    def header_info(self) -> AntsHeader:
        if self.path is None and self.img is not None:
            raise ValueError("Header Info is not available for in memory images.")

        return ants.image_header_info(str(self.path))

    @property
    def metadata(self):
        if self.path is None and self.img is not None:
            raise ValueError("Metadata Info is not available for in memory images.")

        return ants.read_image_metadata(str(self.path))

    @cached_property
    def greyscale_img(self) -> ANTsImage:
        """Returns a greyscale image constructed from the means of the RGB channels, and inverts it (255 becomes 0)"""
        img = self.img

        if img.components <= 1:
            return img

        img_np: npt.NDArray[np.float64] = img.numpy()

        # The mean of RGB values is grayscale (TODO: investigate if taking a single channel as greyscale works better)
        gray_np = np.mean(img_np, axis=-1, dtype=np.float64)

        # Invert. Ants seems to work best with a black background
        inverted = gray_np.max() - gray_np

        return ants.from_numpy(inverted, origin=img.origin, spacing=img.spacing, direction=img.direction)

    @property
    def image_info(self) -> ImageInfo:
        return {
            "shape": self.img.shape,
            "physical_shape": self.img.physical_shape,
            "spacing": self.img.spacing,
            "origin": self.img.origin,
            "direction": self.img.direction,
        }

    def run_qupath_script(self, script_name: str, out_filename: str = "") -> ANTsImage:
        """
        Run a qupath script that takes this image as an argument (`args[0]` in the groovy script), and creates an output image, which we read.
        """
        if self.path is None:
            raise ValueError("Scripts not available for in memory images.")

        print(os.getcwd())
        home = Path.home()
        quPath_dir = home / "AppData" / "Local" / "QuPath-0.7.0"

        cwd = Path(os.getcwd())
        out_path = cwd / f"{out_filename or self.path}-{script_name}.tif"
        project = cwd / "HnE" / "project"

        args = [
            quPath_dir / "QuPath-0.7.0 (console).exe",
            "script",
            "--project",
            project / "project.qpproj",
            "--image",
            self.path.name,
            "--args",
            out_path,
            project / "scripts" / f"{script_name}.groovy",
        ]

        subprocess.run(
            [str(arg) for arg in args],
            cwd=quPath_dir,
        )

        img: ANTsImage = ants.image_read(out_path)  # type: ignore
        self.maps[script_name] = img
        return img

    def svs_read(self, path: Path) -> ANTsImage:
        level = self.level
        if level == 0:
            print("Loading the image with no downscaling. Your computer may explode!")
            return ants.image_read(str(self.path), *self.args, **self.kwargs)

        slide = tiffslide.TiffSlide(path)

        if level >= len(slide.level_downsamples):
            raise IndexError(f"Selected level not available, select from {list(range(len(slide.level_downsamples)))}")

        target_dims = slide.level_dimensions[level]

        rgba_image = slide.read_region((0, 0), level, target_dims)

        # Convert RGBA to RGB, then to a numpy array
        rgb_image = rgba_image.convert("RGB")
        np_img = np.array(rgb_image)

        # We have (Y, X, C), ANTs expects (X, Y, C)
        np_img = np.transpose(np_img, (1, 0, 2))
        np_img = np.ascontiguousarray(np_img)

        header = self.header_info

        ants_img = ants.from_numpy(
            np_img,
            spacing=header["spacing"][:2],
            origin=header["origin"][:2],
            direction=header["direction"][:2, :2],
            has_components=True,
        )

        return ants_img

    def rotate(self, deg: int) -> ANTsImage:
        img = self.img
        return ants.from_numpy(
            np.array(Image.fromarray(img.numpy()).rotate(deg)),
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction,
        )

    def __repr__(self):
        return ANTsImage.__repr__(self.img)


def ants_init(dicom_params: DicomParams, hist_params: HistParams, reg_params: RegParams):
    mri_slices = dicom_params["slices"]

    hist_slices = hist_params["slices"]

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(list(hist_slices), hist_params, return_reg_dict=False)
    else:
        hist_allocation = allocate_hists(list(hist_params["slices"]), hist_params["slide_3_mode"])

    #print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

    base_out_path = Path("out" / reg_params["out_prefix"])

    for mri_key, mri_img in mri_slices.items():

        # Make any (X,Y,1) images (X,Y)
        mri_zero: ANTsImage = ants.slice_image(mri_img.img, axis=-1, idx=0)  # type: ignore

        mri_processed = prepare_mri(mri_zero)
        mri_processed.to_file(ensure_path_exists(base_out_path / f"mri-{mri_img.path.name}-processed.nii.gz"))

        for slice_details in hist_allocation[mri_key]:
            hist_zero = slice_details["img"].greyscale_img
            hist_out_path = base_out_path / slice_details["img"].path.name

            hist_processed = prepare_hist(hist_zero, slice_details, mri_zero)

            registered: RegistrationDict = ants.registration(
                fixed=mri_processed, moving=hist_processed, type_of_transform=reg_params["type_of_transform"]
            )
            # registered["warpedmovout"].to_file(f"out/coloc{mri_key}-{str(slice_details['img'].path.name)}.nii.gz")

            transform_original_hist(
                hist_zero, slice_details, mri_zero, registered["fwdtransforms"], out_path=hist_out_path
            )

            if "maps" in slice_details:
                process_maps(slice_details, mri_zero, registered["fwdtransforms"], out_path=hist_out_path)

            create_checkerboard(
                mri_processed,
                registered["warpedmovout"],
                squares=(8, 8),
                filename=ensure_path_exists(hist_out_path / f"checkerboard.tif"),
            )


def prepare_hist(
    hist: ANTsImage, slice_details: HistSlicesDict, mri: ANTsImage, resample: bool | None = True
) -> ANTsImage:
    if "crop" in slice_details:
        hist = ants.crop_indices(hist, slice_details["crop"][0], slice_details["crop"][1])

    hist_rotated = LazyAntsImage(hist).rotate(slice_details["rotation"])
    # Calculate the physical space of the MRI
    mri_phys_x = mri.shape[0] * mri.spacing[0]  # 176 * 0.1 = 17.6mm
    mri_phys_y = mri.shape[1] * mri.spacing[1]  # 176 * 0.1 = 17.6mm

    # Calculate the spacing needed to match to the physical size of the MRI.
    # i.e How big do the hist pixels need to be to match the phyiscal size of the MRI.
    # new spacing = target physical space (MRI size) / histology resolution
    new_spacing_x = mri_phys_x / hist_rotated.shape[0]
    new_spacing_y = mri_phys_y / hist_rotated.shape[1]

    hist_rotated.set_spacing((new_spacing_x, new_spacing_y))
    hist_rotated.set_origin(mri.origin)
    hist_rotated.set_direction(mri.direction)

    if resample:
        return ants.resample_image_to_target(hist_rotated, mri)  # type: ignore

    # hist_rotated.to_file(f"out/hist_processed-{str(slice_details['img'].path.name)}.nii.gz")

    return hist_rotated


def prepare_mri(mri: ANTsImage) -> ANTsImage:
    mri_bias_corrected = ants.abp_n4(mri)
    # mri_bias_corrected = mri
    mri_thresholded = brain_extraction_threshold(mri_bias_corrected)

    return mri_thresholded


def transform_original_hist(
    hist_zero: ANTsImage, slice_details: HistSlicesDict, mri_zero: ANTsImage, transform: list[str], out_path: Path
) -> ANTsImage:
    """
    Apply the transform from the MRI-histology registration to the original high res histology image.
    We use a downscaled version during registration to improve registration quality, but we can then apply that transform
    to the original for visualisation purposes.
    """
    processed = prepare_hist(hist_zero, slice_details, mri_zero, resample=False)

    transformed: ANTsImage = ants.apply_transforms(
        transformlist=transform, fixed=processed, moving=processed, interpolator="nearestNeighbor"
    )  # type: ignore

    transformed.to_file(
        ensure_path_exists(out_path / f"transformed.tif"),
    )

    return transformed


def process_maps(
    slice_details: HistSlicesDict, mri_zero: ANTsImage, transform: list[str], out_path: Path
) -> dict[str, ANTsImage]:
    if "maps" not in slice_details:
        raise AttributeError(f"Slice {slice_details['img'].path.name} has no associated maps.")

    ret_dict: dict[str, ANTsImage] = {}

    for map_name, map_img in slice_details["maps"].items():
        map_processed = prepare_hist(map_img, slice_details, mri_zero, resample=False)

        map_transformed: ANTsImage = ants.apply_transforms(
            transformlist=transform,
            fixed=map_processed,
            moving=map_processed,
            interpolator="genericLabel",
        )  # type: ignore

        # print(f"""
        #     {mri_zero.spacing=}
        #     {mri_zero.origin=}
        #     {mri_zero.direction=}
        #     {map_transformed.spacing=}
        #     {map_transformed.origin=}
        #     {map_transformed.direction=}
        # """)

        map_transformed.to_file(ensure_path_exists(out_path / f"{map_name}_map.tif"))
        map_transformed.to_file(ensure_path_exists(out_path / f"{map_name}_map.nii.gz"))

        ret_dict[map_name] = map_transformed

    return ret_dict


def allocate_hists(hists: list[HistSlicesDict], slide_3_mode: Literal[0, 1, 2, 3]) -> AllocatedHists[HistSlicesDict]:
    """Assign each hist slide an MRI slide to be coregistered with (assuming 5 hist slides and 2 MRI slides)"""
    dict = {}
    match slide_3_mode:
        case 0:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[3:5]}
        case 1:
            dict: AllocatedHists = {"mri_1": hists[0:3], "mri_2": hists[3:5]}
        case 2:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[2:5]}
        case 3:
            dict: AllocatedHists = {"mri_1": hists[0:3], "mri_2": hists[2:5]}
    return dict


@overload
def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, return_reg_dict: Literal[True]
) -> AllocatedHists[RegistrationDict]:
    pass


@overload
def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, return_reg_dict: Literal[False]
) -> AllocatedHists[HistSlicesDict]:
    pass


def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, return_reg_dict: bool = False
) -> AllocatedHists[T]:
    """
    Register hist slices against others. Returns a dict with hist slides allocated to MRI slides, optionally including the registration info. See `allocate_hists`
    """

    registered: AllocatedHists = {"mri_1": [], "mri_2": []}

    fixed = hists[hist_params.get("fixed_image", 2)]["img"].greyscale_img
    moving_dict = allocate_hists(list(hists), hist_params["slide_3_mode"])

    for mri_key, hist_dicts in moving_dict.items():
        for i, hist_dict in enumerate(hist_dicts):

            moving = hist_dict["img"].greyscale_img

            reg: RegistrationDict = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNRA")

            registered[mri_key].append(
                reg
                if return_reg_dict
                else {"img": LazyAntsImage(reg["warpedmovout"]), "rotation": hist_dict["rotation"]}
            )
            gc.collect()
    return registered


def brain_extraction_threshold(mri: LazyAntsImage | ANTsImage) -> ANTsImage:

    if isinstance(mri, LazyAntsImage):
        img = mri.img
    elif isinstance(mri, ANTsImage):
        img = mri
    else:
        raise ValueError("mri must be a LazyAntsImage, or an AntsImage")

    np_arr = img.numpy()

    from skimage.filters import threshold_otsu
    from skimage import morphology, measure
    from skimage.morphology import ball, disk

    if np_arr.ndim == 2:
        footprint_fn = disk
    elif np_arr.ndim == 3:
        footprint_fn = ball
    else:
        raise ValueError("mri must be either 2D or 3D")

    thresh = threshold_otsu(np_arr)
    binary = np_arr > thresh

    eroded = morphology.erosion(binary, footprint_fn(3))

    labeled = measure.label(eroded)

    # Keep only the largest region in the binary image (the brain itself)
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda x: x.area)

    # Create the final mask using the largest region
    mask = np.zeros_like(np_arr, dtype=bool)
    mask[labeled == largest.label] = True

    dilated_mask = morphology.dilation(mask, footprint_fn(6))

    brain = np_arr * dilated_mask

    return ants.new_image_like(img, brain)

def create_checkerboard(
    img1: ANTsImage, img2: ANTsImage, squares: tuple[int, int] | None = None, filename: str | None = None
) -> sitk.Image:
    img = sitk.CheckerBoard(ants.to_sitk(img1), ants.to_sitk(img2), squares)
    ants.from_sitk(img).to_file(filename)
    return img

def ensure_path_exists(str_or_path: str | Path) -> str:
    """Mainly for use with ants filenames. Recursively create any necessary folders for the inputted filename."""
    path = Path(str_or_path)

    path = path.parent if bool(path.suffix) else path

    Path.mkdir(path, parents=True, exist_ok=True)
    return str(str_or_path)


if __name__ == "__main__":
    dicom: DicomParams = {
        "slices": {
            "mri_1": LazyAntsImage(Path("AV_GBM_PK_AV_GBM_PK_GCBA5.23O_SC1__E2_P1") / "MRIm08.dcm", dimension=3),
            "mri_2": LazyAntsImage(Path("AV_GBM_PK_AV_GBM_PK_GCBA5.23O_SC1__E2_P1") / "MRIm09.dcm", dimension=3),
        }
    }
    hist: HistParams = {
        "loc_within": False,
        "fixed_image": 2,
        "slide_3_mode": 2,
        "slices": (
            {
                "img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S1.svs"),
                "rotation": 100,
                "maps": {"cell_count": ants.image_read("240920_GCBA_23o_HnE20x_S1.svs-cell-count.tif")},
            },
            {
                "img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S2.svs"),
                "rotation": 100,
                "crop": ((0, 0), (2816, 3072)),
            },
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S3.svs"), "rotation": 100},
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S4.svs"), "rotation": 100},
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S5.svs"), "rotation": 100},
        ),
    }
    reg: RegParams = {"type_of_transform": "SyNRA", "out_prefix": Path("1")}
    ants_init(dicom_params=dicom, hist_params=hist, reg_params=reg)
