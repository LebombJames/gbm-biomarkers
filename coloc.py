from __future__ import annotations
import ants
import tiffslide
import os
from pydicom import dcmread
import numpy as np

# from vedo import Volume, show
import vedo
from typing import overload, Literal, cast, no_type_check, Optional, Callable, Any
from typing_extensions import TypeIs
from pathlib import Path
from __types import *
import gc
from functools import cached_property

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import subprocess
import os
import docker, docker.errors, docker.models.containers
from PIL import Image
import SimpleITK as sitk

import ants.config
ants.config.set_ants_deterministic(True, 123)


class LazyAntsImage:
    """
    A wrapper around Path that lazy loads an AntsImage when called, or returns a cached version.
    """

    def __init__(
        self,
        pathOrImg: Path | ants.ANTsImage,
        cell_count: ants.ANTsImage | None = None,
        level: int = 2,
        *args,
        **kwargs,
    ) -> None:
        self.level = level
        self.args = args
        self.kwargs = kwargs

        if isinstance(pathOrImg, Path):
            self.path = pathOrImg
        elif isinstance(pathOrImg, ants.ANTsImage):
            self.img = pathOrImg
        else:
            raise ValueError("Neither a Path or an AntsImage were provided.")

        if cell_count is not None:
            self.cell_count = cell_count

    @cached_property
    def img(self) -> ants.ANTsImage:
        """Return the cached Ants image, or load it from the path if not loaded yet."""

        if self.path is None:
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately before
            # this whole function is even called.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        print(f"Loading image: {self.path}")

        return (
            self.svs_read(self.path, self.level)
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
    def greyscale_img(self) -> ants.ANTsImage:
        """Returns a greyscale image constructed from the means of the RGB channels, and inverts it (255 becomes 0)"""
        img = self.img

        if img.components <= 1:
            return img

        img_np: npt.NDArray[np.float64] = img.numpy()

        # The mean of RGB values is grayscale
        gray_np = np.mean(img_np, axis=-1, dtype=np.float64)

        # Invert. Ants seems to work best with a black background
        inverted = gray_np.max() - gray_np

        return ants.from_numpy(inverted, origin=img.origin, spacing=img.spacing, direction=img.direction)

    @cached_property
    def cell_count(self) -> ants.ANTsImage:
        """
        Call QuPath and produce an image where the intensity of each pixel corresponds to the number of cells in each tile.

        ### Can take upwards of 10 minutes!
        """
        if self.path is None:
            raise ValueError("Cell Counting is not available for in memory images.")

        print(os.getcwd())
        home = Path.home()
        quPath_dir = home / "AppData" / "Local" / "QuPath-0.7.0"

        cwd = Path(os.getcwd())
        out_path = cwd / f"{self.path}-cell-count.tif"
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
            project / "scripts" / "cell count mask.groovy",
        ]

        subprocess.run(
            [str(arg) for arg in args],
            cwd=quPath_dir,
        )

        return ants.image_read(out_path)

    def to_png(self):
        return ants.image_write(self.img.astype("uint8"), filename=f"{self.path}.png")

    def np_func(self, op: Callable[[npt.NDArray], npt.NDArray], single_components: bool = False) -> ants.ANTsImage:
        """
        Helper to perform a numpy array function on an Ants image.

        Usage: `img.np_func(lambda x: np.rot90(x))`, where `x` is the numpy array of the image.

        """
        return ants.new_image_like(self.img, op(self.img.numpy(single_components)))

    def svs_read(self, path: Path, level: int = 1) -> ants.ANTsImage:
        if level == 0:
            return ants.image_read(str(path))

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

    def rotate(self, deg: int) -> ants.ANTsImage:
        img = self.img
        return ants.from_numpy(
            np.array(Image.fromarray(img.numpy()).rotate(deg)),
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction,
        )

    def __repr__(self):
        return ants.ANTsImage.__repr__(self.img)


def load_dicom_to_vedo(directory_path: str):
    # 1. Grab ONLY the .dcm files, ignoring everything else
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".dcm")]

    if not files:
        raise ValueError("No .dcm files found in the specified directory.")

    # 2. Read them with pydicom
    slices = [dcmread(f) for f in files]

    # 3. Sort them strictly by their Z-axis position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 4. Stack into a 3D Numpy Array
    # pydicom arrays are stacked as (Z, Y, X)
    volume_data = np.stack([s.pixel_array for s in slices])

    volume_data = volume_data * 255.0 / volume_data.max()
    print(volume_data[1])

    # 5. Extract the spacing
    dy, dx = slices[0].PixelSpacing  # Rows, Columns
    dz = slices[0].SliceThickness

    # 6. Create the Vedo Volume directly from the Numpy array
    vol = vedo.Volume(volume_data)

    # Apply the spacing to ensure the volume isn't distorted
    # We pass it as [dz, dy, dx] to match the numpy array's (Z, Y, X) shape
    vol.spacing([dz, dy, dx])

    return vol


def render_dicom_volume():
    dicom_dir = "GCBA5.23O_SC1__E2"  # Ensure this is a string!

    # Load the volume
    vol = load_dicom_to_vedo(dicom_dir)
    print("Volume successfully built from Numpy array!")

    # Style and render
    vmin, vmax = 1, 68
    vol.cmap("bone")
    vol.alpha([(0, 0.0), (5, 0.0), (20, 1.0), (25, 1.0)])
    vol.mode(1)
    vedo.show(vol, axes=1, bg="white")


def explore_3D_array(arr: npt.NDArray):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.2)

    # Set the starting slice (e.g., the middle of the volume)
    initial_slice = 0

    im = ax.imshow(arr[:, :, initial_slice], cmap="gray")
    ax.set_title(f"Slice: {initial_slice}")
    ax.axis("off")

    # 3. Create a dedicated set of axes just for the slider
    # Dimensions are [left, bottom, width, height] in figure percentages
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

    # 4. Initialize the native Matplotlib Slider
    slice_slider = Slider(
        ax=ax_slider, label="Z-Slice", valmin=0, valmax=arr.shape[2] - 1, valinit=initial_slice, valstep=1
    )

    def update(val):
        z_index = int(slice_slider.val)

        # Update the existing image data instead of creating a new plot
        im.set_data(arr[:, :, z_index])
        ax.set_title(f"Slice: {z_index}")

        # Tell matplotlib to redraw the figure
        fig.canvas.draw_idle()

    # 6. Connect the slider movement to the update function
    slice_slider.on_changed(update)

    # 7. CRITICAL FIX: Attach the slider to the figure so it isn't garbage collected
    # fig.slider_obj = slice_slider

    plt.show()


def ants_init(dicom_params: DicomParams, hist_params: HistParams):
    mri_slices = dicom_params["slices"]

    hist_slices = hist_params["slices"]

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(list(hist_slices), hist_params, False)
    else:
        hist_allocation = allocate_hists(list(hist_params["slices"]), hist_params["slide_3"])

    for mri_key, mri_img in mri_slices.items():

        mri_zero: ants.ANTsImage = ants.slice_image(mri_img.img, axis=-1, idx=0)  # type: ignore

        mri_processed = prepare_mri(mri_zero)

        for hist_dict in hist_allocation[mri_key]:
            hist_zero = hist_dict["img"].greyscale_img

            hist_processed = prepare_hist(hist_zero, hist_dict, mri_zero)

            ants.image_write(
                hist_processed.astype("uint8"),
                filename=f"out/hist_processed-{str(hist_dict['img'].path.name)}.png",
            )

            # cell_count = LazyAntsImage(hist_dict["img"].cell_count)

            # cells_rotated = cell_count.rotate(hist_dict["rotation"])

            # print(
            #     f"{mri_processed.spacing=}",
            #     f"{mri_processed.shape=}",
            #     f"{mri_processed.origin=}",
            #     f"{mri_processed.direction=}",
            #     f"{hist_processed.spacing=}",
            #     f"{hist_processed.shape=}",
            #     f"{hist_processed.origin=}",
            #     f"{hist_processed.direction=}",
            # )

            coloc_mri_1: RegistrationDict = ants.registration(
                fixed=mri_processed, moving=hist_processed, type_of_transform="SyNRA"
            )

            hist_zero = LazyAntsImage(hist_zero).rotate(hist_dict["rotation"])
            ants.image_write(
                hist_zero.astype("uint8"),
                filename=f"out/hist_zero-{str(hist_dict['img'].path.name)}.png",
            )

            transformed = cast(
                ants.ANTsImage,
                ants.apply_transforms(transformlist=coloc_mri_1["fwdtransforms"], fixed=hist_zero, moving=hist_zero),
            )

            # cells_transformed = ants.apply_transforms(
            #     transformlist=coloc_mri_1["fwdtransforms"], fixed=cells_rotated, moving=cells_rotated
            # )

            # print(coloc_mri_1["warpedmovout"])

            ants.image_write(
                coloc_mri_1["warpedmovout"].astype("uint8"),
                filename=f"out/coloc{mri_key}-{str(hist_dict['img'].path.name)}.png",
            )
            ants.image_write(
                transformed.astype("uint8"), filename=f"out/transformed{mri_key}-{str(hist_dict['img'].path.name)}.png"
            )

            ants.image_write(
                ants.from_sitk(checkerboard(mri_processed, coloc_mri_1["warpedmovout"], squares=(16,16))).astype("uint8"),
                filename=f"out/checkerboard{mri_key}-{str(hist_dict['img'].path.name)}.png",
            )

        # ants.image_write(cell_count.astype("uint8"), filename=f"cells.png")
        # ants.image_write(cells_transformed.astype("uint8"), filename="cells_transformed.png")


def prepare_hist(hist: ants.ANTsImage, hist_dict: HistParamsDict, mri: ants.ANTsImage) -> ants.ANTsImage:
    hist_rotated = LazyAntsImage(hist).rotate(hist_dict["rotation"])

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

    hist_downscaled: ants.ANTsImage = ants.resample_image_to_target(hist_rotated, mri)  # type: ignore

    return hist_downscaled


def prepare_mri(mri: ants.ANTsImage) -> ants.ANTsImage:
    mri_bias_corrected = ants.abp_n4(mri)
    mri_thresholded = brain_extraction_threshold(mri_bias_corrected)

    return mri_thresholded


def allocate_hists(hists: list[HistParamsDict], slide_3: Literal[0, 1, 2, 3]) -> AllocatedHists[HistParamsDict]:
    """Assign each hist slide an MRI slide to be coregistered with (assuming 5 hist slides and 2 MRI slides)"""
    dict = {}
    match slide_3:
        case 0:
            dict: AllocatedHists = {"mri_1": hists[0:1], "mri_2": hists[3:4]}
        case 1:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[3:4]}
        case 2:
            dict: AllocatedHists = {"mri_1": hists[0:1], "mri_2": hists[2:4]}
        case 3:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[2:4]}
    return dict


@overload
def register_hist_within(
    hists: list[HistParamsDict], hist_params: HistParams, return_reg_dict: Literal[True]
) -> AllocatedHists[RegistrationDict]:
    pass


@overload
def register_hist_within(
    hists: list[HistParamsDict], hist_params: HistParams, return_reg_dict: Literal[False]
) -> AllocatedHists[HistParamsDict]:
    pass


def register_hist_within(
    hists: list[HistParamsDict], hist_params: HistParams, return_reg_dict: bool = False
) -> AllocatedHists[T]:
    """Register hist slices against others. Returns a dict with hist slides allocated to MRI slides, optionally including the registration info. See `allocate_hists`"""

    registered: AllocatedHists = {"mri_1": [], "mri_2": []}

    fixed = hists[hist_params.get("fixed_image", 2)]["img"].greyscale_img
    moving_dict = allocate_hists(list(hists), hist_params["slide_3"])

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


def stack_hist(hists: list[LazyAntsImage]) -> ants.ANTsImage:
    """Create a 3d volume from hist slices"""
    arrays = [img.img.numpy() for img in hists]

    volume_array = np.stack(arrays, axis=-1)

    # Rebuild metadata to include a 3rd element in the tuple for the Z axis.
    reference = hists[0].img

    slice_thickness = 0.08  # 80 microns
    z_origin = 0.0

    spacing = (*reference.spacing, slice_thickness)
    origin = (*reference.origin, z_origin)

    direction = np.eye(3)
    direction[:2, :2] = reference.direction

    return ants.from_numpy(volume_array, spacing=spacing, origin=origin, direction=direction)


def is_registration_dict(
    dict: AllocatedHists,
) -> TypeIs[AllocatedHists[RegistrationDict]]:
    """Does this AllocatedHist object contain arrays of RegistrationDicts for each MRI slide?"""
    return hasattr(dict["mri_1"][0], "warpedmovout")


def alloc_registration_dict_to_images(
    alloc: AllocatedHists[RegistrationDict],
) -> AllocatedHists[LazyAntsImage]:
    """Convert an AllocatedHist object from containing arrays of RegistrationDicts for each MRI slide to arrays of images"""
    return {
        "mri_1": [LazyAntsImage(reg["warpedmovout"]) for reg in alloc["mri_1"]],
        "mri_2": [LazyAntsImage(reg["warpedmovout"]) for reg in alloc["mri_2"]],
    }


def brain_extraction_nirodents(mri: LazyAntsImage):

    client = docker.from_env()

    templateflow_host = os.path.expanduser("~/.cache/templateflow")
    os.makedirs(templateflow_host, exist_ok=True)

    volumes = {
        os.getcwd(): {"bind": "/data", "mode": "ro"},
        f"{os.getcwd()}/out": {"bind": "/out", "mode": "rw"},
        templateflow_host: {"bind": "/templateflow", "mode": "rw"},
    }

    # Not strictly necessary since run also pulls, but it creates console output when it's done
    image = client.images.pull("nipreps/nirodents:0.2.1")

    try:
        container = cast(
            docker.models.containers.Container,
            client.containers.run(
                image=image,
                command=f"/data/{mri.path.as_posix() if isinstance(mri.path, Path) else ''} -o /out/ -w /out/work",
                volumes=volumes,
                environment={"TEMPLATEFLOW_HOME": "/templateflow"},
                remove=True,
                detach=True,
            ),
        )

        for log_line in container.logs(stream=True):
            print(log_line.decode("utf-8"), end="")

        result = container.wait()
        print(f"\nContainer exited with code: {result['StatusCode']}")

    except docker.errors.ContainerError as e:
        print(e.stderr.decode("utf-8") if e.stderr else e.container.logs().decode("utf-8"))
    except Exception as e:
        raise Exception from e


def brain_extraction_threshold(mri: LazyAntsImage | ants.ANTsImage) -> ants.ANTsImage:

    if isinstance(mri, LazyAntsImage):
        img = mri.img
    elif isinstance(mri, ants.ANTsImage):
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

    # Keep only the largest region in the binary image (the brain)
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda x: x.area)

    # Create the final mask using the largest region
    mask = np.zeros_like(np_arr, dtype=bool)
    mask[labeled == largest.label] = True

    dilated_mask = morphology.dilation(mask, footprint_fn(6))

    brain = np_arr * dilated_mask

    return ants.new_image_like(img, brain)


def checkerboard(img1: ants.ANTsImage, img2: ants.ANTsImage, squares: tuple[int, int] | None = None) -> sitk.Image:
    return sitk.CheckerBoard(ants.to_sitk(img1), ants.to_sitk(img2), squares)


# pix = LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S1.svs")).count_cells
if __name__ == "__main__":
    dicom: DicomParams = {
        "slices": {
            "mri_1": LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm08.dcm", dimension=3),
            "mri_2": LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm09.dcm", dimension=3),
        }
    }
    hist: HistParams = {
        "fixed_image": 2,
        "loc_within": False,
        "slide_3": 3,
        "slices": (
            {
                "img": LazyAntsImage(
                    Path("HnE") / "240920_GCBA_23o_HnE20x_S1.svs",
                    cell_count=ants.image_read("240920_GCBA_23o_HnE20x_S1.svs-cell-count.tif"),
                ),
                "rotation": 100,
            },
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S2.svs"), "rotation": 100},
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S3.svs"), "rotation": 100},
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S4.svs"), "rotation": 100},
            {"img": LazyAntsImage(Path("HnE") / "240920_GCBA_23o_HnE20x_S5.svs"), "rotation": 100},
        ),
    }
    ants_init(dicom_params=dicom, hist_params=hist)

    # mri = ants.dicom_read(str(Path("GCBA5.23O_SC1__E2")))
    #    # mri = ants.abp_n4(mri)
    #     ants.image_write(ants.slice_image(mri, axis=-1, idx=7).astype("uint8"), filename="mri.png")
    #     extracted: ants.ANTsImage = brain_extraction(mri, modality="t2", verbose=True)

    #     ants.get_mask(extracted)
    #     explore_3D_array(extracted.numpy() * 255)

    # a = ants.image_read("C:/Users/sam_m/Documents/Glasgow/Diss/py/240920_GCBA_23o_HnE20x_S1.svs-cell-count.tif")
    # ants.image_write(a.astype("uint8"), filename="cells.png")
    # mri = LazyAntsImage(Path("dataset") / "sub-1" / "anat" / "sub-1_mouse_t2_centered.nii.gz")
    # print(mri.path.as_posix() if isinstance(mri.path, Path) else "")
    # print(mri.img.numpy().ndim)
    # a = brain_extraction_threshold(mri)
    # ants.image_write(a, filename="threshold.nii.gz")
