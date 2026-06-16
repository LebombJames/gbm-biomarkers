from __future__ import annotations
import ants
from pydicom import datadict
from ants import ANTsImage
import tiffslide
import numpy as np
from PIL import Image
import SimpleITK as sitk

from skimage.filters import threshold_otsu
from skimage import morphology, measure
from skimage.morphology import ball, disk
from skimage.color import rgb2hed, hed2rgb
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import gc
from typing import overload, Literal, Any, cast
from typing_extensions import TypeIs
from pathlib import Path
from functools import cached_property
import subprocess
import os
from collections import defaultdict
import math
import asyncio

from __types import *

import ants.config
import random

seed = 123
ants.config.set_ants_deterministic(True, seed)
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
np.random.seed(seed)
random.seed(seed)


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

        self.maps: dict[str, ANTsImage] = {}

        if isinstance(path_or_img, Path):
            self.path = path_or_img
        elif isinstance(path_or_img, ANTsImage):
            self.img = path_or_img
        else:
            raise TypeError("Neither a Path or an AntsImage were provided.")

    @cached_property
    def img(self) -> ANTsImage:
        """Return the cached Ants image, or load it from the path if not loaded yet."""

        if self.path is None:
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately before
            # this whole function is even called.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        print(f"Loading image: {self.path}")

        return self.svs_read(self.path) if self.is_hist else ants.image_read(str(self.path), *self.args, **self.kwargs)

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

        metadata: dict[str, Any] = ants.read_image_metadata(str(self.path))

        translated = {}
        for key, value in metadata.items():
            # Check if the key matches the "XXXX|YYYY" DICOM tag pattern
            split = key.split("|")
            if len(split) != 2:
                # Include any other keys that don't fit the pattern as a fallback
                translated[key] = value
                continue

            try:
                # Pydicom can take a tuple of the key as (XXXX, YYYY)
                readable_name = datadict.keyword_for_tag(split)  # type: ignore

                new_key = readable_name if readable_name else key
                translated[new_key] = value

            except:
                # Fallback just in case the lookup fails
                translated[key] = value  #

        return translated

    def greyscale_img(self, mode: GreyscaleModes = "mean") -> ANTsImage:
        """
        Returns a greyscale image constructed from the means of the RGB channels, and inverts it (255 becomes 0)

        Args:
            mode: How the greyscale is calculated. Accepts "red", "green", "blue", to use only those channels, "mean" to average all three, or a set containing any combination of "red", "green", and "blue" to average those. Defaults to "mean".
        """
        img = self.img

        if img.components <= 1:
            return img

        # In an AntsImage, the 0th colour channel is red, the 1st green, and 2nd blue.
        colour_map: dict[Literal["red", "green", "blue"], Literal[0, 1, 2]] = {"red": 0, "green": 1, "blue": 2}

        img_np: npt.NDArray[np.float32] = img.numpy()

        if mode == "mean" or mode == set(colour_map.keys()):
            # The mean of RGB values is grayscale
            gray_np = np.mean(img_np, axis=-1, dtype=np.float64)

            # Invert. Ants seems to work best with a black background
            inverted = gray_np.max() - gray_np
        elif mode in ["h", "e", "h&e"]:
            mode = cast(Literal["h", "e", "h&e"], mode)
            ihc_hed = rgb2hed(img.numpy())

            null = np.zeros_like(ihc_hed[:, :, 0])
            ihc_h = ihc_hed[:, :, 0]  # hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            ihc_e = ihc_hed[:, :, 1]  # hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            ihc_d = ihc_hed[:, :, 2]

            if mode == "h&e":
                he_image = hed2rgb(np.stack((ihc_hed[:, :, 0], ihc_hed[:, :, 1], null), axis=-1))
                gray_np = np.mean(he_image, axis=-1, dtype=np.float64)

                inverted = gray_np.max() - gray_np
            elif mode == "e":
                inverted = ihc_e.max() - ihc_e
            elif mode == "h":
                inverted = ihc_h.max() - ihc_h
        elif isinstance(mode, str):  # "red", "green", "blue"
            try:
                colour_idx = colour_map[cast(Literal["red", "green", "blue"], mode)]
            except KeyError as e:
                raise KeyError(f"{mode} is not a valid greyscale mode") from e

            colour_np = ants.split_channels(img)[colour_idx].numpy()
            inverted = colour_np.max() - colour_np

        elif isinstance(mode, set):
            try:
                channels_idx = {colour_map[colour] for colour in mode}
            except KeyError as e:
                raise KeyError(f"{mode} is not a valid greyscale mode") from e

            split_channels = ants.split_channels(img)
            channels = [split_channels[channel_idx] for channel_idx in channels_idx]
            merged = ants.merge_channels(channels).numpy()

            merged_np = np.mean(merged, axis=-1, dtype=np.float64)

            inverted = merged_np.max() - merged_np

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

    def run_qupath_script(
        self, project_path: Path, script_name: str, out_filename: str = "", script_args: list[str] | None = None
    ) -> ANTsImage:
        """
        Run a qupath script that takes this image as an argument (`args[0]` in the groovy script),
        and creates an output image, which we read and return
        """
        if self.path is None:
            raise ValueError("Scripts not available for in memory images.")

        if not script_args:
            script_args = []

        processed_args = []
        for arg in script_args:
            processed_args.append("--args")
            processed_args.append(arg)

        home = Path.home()
        quPath_dir = home / "AppData" / "Local" / "QuPath-0.7.0"

        cwd = Path(os.getcwd())
        out_path = cwd / f"{out_filename or self.path.name}-{script_name}.tif"
        project = cwd / project_path

        args = [
            quPath_dir / "QuPath-0.7.0 (console).exe",
            "script",
            "--project",
            project / "project.qpproj",
            "--image",
            self.path.name,
            "--args",
            out_path,
            *processed_args,
            project / "scripts" / f"{script_name}.groovy",
        ]

        print([str(arg) for arg in args])

        try:
            subprocess.run(
                [str(arg) for arg in args],
                cwd=quPath_dir,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"QuPath Failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise Exception from e

        img: ANTsImage = ants.image_read(str(out_path))  # type: ignore
        self.maps[script_name] = img
        return img

    async def run_qupath_script_async(self, script_name: str, out_filename: str = "") -> ANTsImage:
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

        qupath_exe = (quPath_dir / "QuPath-0.7.0 (console).exe",)

        args = [
            qupath_exe,
            "script",
            "--project",
            project / "project.qpproj",
            "--image",
            self.path.name,
            "--args",
            out_path,
            project / "scripts" / f"{script_name}.groovy",
        ]

        str_args = [str(arg) for arg in args]

        # 1. Start the subprocess
        process = await asyncio.create_subprocess_exec(
            str(qupath_exe),
            *str_args[1:],
            cwd=quPath_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 2. Define a helper to stream lines as they arrive
        async def stream_output(stream: asyncio.StreamReader, prefix: str):
            while True:
                line = await stream.readline()
                if not line:
                    break  # EOF reached

                # Print live to the console.
                # end="" is used because `line` already contains the newline character.
                print(f"{prefix}{line.decode()}", end="")

        # 3. Read stdout, read stderr, and wait for the process to exit concurrently
        if process.stderr is None or process.stdout is None:
            raise AttributeError("process has no stdout or stderr somehow!")

        await asyncio.gather(
            stream_output(process.stdout, "[QuPath INFO] "),
            stream_output(process.stderr, "[QuPath ERR]  "),
            process.wait(),
        )

        # 4. Check exit status
        if process.returncode != 0:
            raise RuntimeError(f"QuPath script failed with return code {process.returncode}.")

        # 5. Read the image asynchronously
        loop = asyncio.get_running_loop()
        img: ANTsImage = await loop.run_in_executor(None, ants.image_read, str(out_path))  # type: ignore

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
            has_components=img.has_components,
        )

    def __repr__(self):
        return ANTsImage.__repr__(self.img)


def ants_init(dicom_params: DicomParams, hist_params: HistParams, reg_params: RegParams):
    mri_slices = dicom_params["slices"]
    hist_slices = hist_params["slices"]

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(hist_slices, hist_params, dicom_params, return_reg_dict=False)
    else:
        hist_allocation = allocate_hists(hist_slices, dicom_params)

    # print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

    base_out_path = Path("out") / reg_params["out_prefix"]

    for mri_key, mri_dict in mri_slices.items():

        mri_img = mri_dict["img"]

        # Make any (X,Y,1) images (X,Y)
        mri_zero: ANTsImage = ants.slice_image(mri_img.img, axis=-1, idx=0)  # type: ignore
        mri_zero.set_direction(np.eye(2))

        mri_prepared_dict = prepare_mri(mri_zero)
        mri_processed = mri_prepared_dict["img"]
        mri_mask = mri_prepared_dict["mask"]  # type: ignore
        mri_processed.astype("uint8").to_file(ensure_path_exists(base_out_path / f"{mri_key}-processed.png"))

        for slice_details in hist_allocation[mri_key]:
            hist_zero = slice_details["img"].greyscale_img(hist_params["greyscale_type"])
            hist_out_path = base_out_path / slice_details["img"].path.name / mri_key

            hist_processed = prepare_hist(
                hist_zero, slice_details, mri_processed, mri_mask, center=False, resample=True
            )
            hist_img = hist_processed["img"]
            hist_mask = hist_processed["mask"]  # type: ignore

            # Sanity check
            # hist_mask = ants.copy_image_info(hist_img, hist_mask)
            # mri_mask = ants.copy_image_info(mri_processed, mri_mask)

            affine_init = None
            if reg_params["use_initial_transform"]:
                affine_init = ants.affine_initializer(fixed_image=mri_processed, moving_image=hist_img)

            hist_img.astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_hist.png"))
            (hist_mask * 255).astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_mask.png"))

            print(f"""
                {hist_img.origin=}
                {mri_processed.origin=}
                {hist_mask.origin=}
                {mri_mask.origin=}
                  """)

            print("MRI Mask unique values:", np.unique(mri_mask.numpy()))
            print("Hist Mask unique values:", np.unique(hist_mask.numpy()))

            registered: RegistrationDict = ants.registration(
                fixed=mri_processed,
                moving=hist_img,
                mask=mri_mask,
                moving_mask=hist_mask,
                initial_transform=affine_init,
                **reg_params,
            )
            print(f"{registered['warpedmovout'].shape=}, {mri_processed.shape=}")
            registered["warpedmovout"].astype("uint8").to_file(ensure_path_exists(hist_out_path / f"coloc.png"))
            registered["warpedmovout"].to_file(ensure_path_exists(hist_out_path / f"coloc.nii.gz"))

            transform_original_hist(
                hist_zero, slice_details, mri_processed, mri_mask, registered["fwdtransforms"], out_path=hist_out_path
            )

            if slice_details["maps"]:
                maps = process_maps(
                    slice_details,
                    registered["fwdtransforms"],
                    mri_processed,
                    mri_key=mri_key,
                    dicom_params=dicom_params,
                    out_path=hist_out_path,
                )

                for map_name, map_dict in maps.items():

                    plot_roi_intensity(
                        map_dict["img"],
                        mri_processed,
                        roi_hist=((40, 100), (115, 140)),
                        roi_mri=((40, 100), (115, 140)),
                        out_path=hist_out_path / "maps" / f"{map_name} overview.png",
                        mi=map_dict["mutual_info"],
                    )

            create_checkerboard(mri_processed, registered["warpedmovout"], squares=(8, 8), out_path=hist_out_path)

            create_hist_volume(registered["warpedmovout"], mri_key, dicom_params, out_path=hist_out_path)


@overload
def prepare_hist(
    hist: ANTsImage,
    slice_details: HistSlicesDict,
    mri: ANTsImage,
    mri_mask: ANTsImage | None = None,
    *,
    threshold: Literal[True] = True,
    center: bool = True,
    resample: bool = True,
    interp: str = "linear",
    out_path: Path | None = None,
) -> ThresholdDict:
    pass


@overload
def prepare_hist(
    hist: ANTsImage,
    slice_details: HistSlicesDict,
    mri: ANTsImage,
    mri_mask: ANTsImage | None = None,
    *,
    threshold: Literal[False],
    center: bool = True,
    resample: bool = True,
    interp: str = "linear",
    out_path: Path | None = None,
) -> ANTsImage:
    pass


def prepare_hist(
    hist: ANTsImage,
    slice_details: HistSlicesDict,
    mri: ANTsImage,
    mri_mask: ANTsImage | None = None,
    *,
    threshold: bool = True,
    center: bool = True,
    resample: bool = True,
    interp: str = "linear",
    out_path: Path | None = None,
) -> ANTsImage | ThresholdDict:
    if "crop" in slice_details:
        hist = ants.crop_indices(hist, slice_details["crop"][0], slice_details["crop"][1])

    shape = hist.shape
    # Pythagoras: determine the diagonal length of the image
    diagonal = math.ceil(math.sqrt(shape[0] ** 2 + shape[1] ** 2))

    # Pad the image equally on all sides to fit the diagonal
    # This ensures the rotation has enough space on all sides, so that the histology doesn't get clipped
    # The diagnoal is the max possible width the image can be when rotating, so if we accomodate for that, the image will never clip
    # We calculate the difference between the diagonal and the current dimensions because ants adds the required size
    pad_x = (diagonal - shape[0]) // 2
    pad_y = (diagonal - shape[1]) // 2

    padded: ANTsImage = ants.pad_image(hist, pad_width=[(pad_x, pad_x), (pad_y, pad_y)], value=0)  # type: ignore
    hist_rotated = LazyAntsImage(padded).rotate(slice_details["rotation"])
    if out_path:
        hist_rotated.astype("uint8").to_file(ensure_path_exists(out_path / "hist-rotated.png"))

    if threshold:
        thresholded = threshold_img(hist_rotated, destructive=False)
        return prepare_hist_thresholding(thresholded, mri, mri_mask, center, resample, interp, out_path)
    else:
        scaled_img = scale_and_align_to_ref(hist_rotated, mri, interp)

    if resample:
        final_img: ANTsImage = ants.resample_image_to_target(scaled_img, mri, interp_type=interp)  # type: ignore

        if out_path:
            final_img.astype("uint8").to_file(ensure_path_exists(out_path / "downscaled.png"))
    else:
        final_img = scaled_img

    return final_img


def prepare_hist_thresholding(
    threshold_dict: ThresholdDict,
    mri: ANTsImage,
    mri_mask: ANTsImage | None,
    center: bool = False,
    resample: bool = True,
    interp: str = "linear",
    out_path: Path | None = None,
) -> ThresholdDict:

    scaled_img = scale_and_align_to_ref(threshold_dict["img"], mri, interp)
    scaled_mask = scale_and_align_to_ref(threshold_dict["mask"], mri, interp)
    if out_path:
        (scaled_mask * 255).astype("uint8").to_file(ensure_path_exists(out_path / "scaled-mask.png"))

    if center:
        if mri_mask is None:
            raise ValueError("An MRI mask must be provided perform centering.")

        centered = align_centers_physically(mri, scaled_img, mri_mask, scaled_mask)
        fullres_img = centered["img"]
        fullres_mask = centered["mask"]
        print(f"{fullres_img.origin=}, {fullres_mask.origin=}")
    else:
        fullres_img = scaled_img
        fullres_mask = scaled_mask

    if resample:
        final_img: ANTsImage = ants.resample_image_to_target(fullres_img, mri, interp_type=interp)  # type: ignore
        # This needs to be float32 or it outputs a black image??
        final_mask: ANTsImage = ants.resample_image_to_target(
            fullres_mask.astype("float32"), mri, interp_type="genericLabel"
        )  # type: ignore

        # final_img = ants.resample_image(fullres_img, resample_params=mri.shape[0:2], use_voxels=True)
        # final_mask = ants.resample_image(fullres_mask, resample_params=mri.shape[0:2], use_voxels=True)  # type: ignore
        if out_path:
            final_img.astype("uint8").to_file(ensure_path_exists(out_path / "downscaled.png"))
            (final_mask * 255).astype("uint8").to_file(ensure_path_exists(out_path / "mask-downsclaed.png"))
    else:
        final_img = scaled_img
        final_mask = scaled_mask  # type: ignore

    out: ThresholdDict = {
        "mask": final_mask,
        "img": final_img,
        "region": threshold_dict["region"],
    }

    return out


def align_centers_physically(
    mri: ANTsImage, hist: ANTsImage, mri_mask: ANTsImage, hist_mask: ANTsImage
) -> dict[Literal["img", "mask"], ANTsImage]:
    """
    Shifts the moving image in physical space so its center overlaps the fixed image.
    Does not pad, resample, or alter the underlying pixel arrays.
    """

    mri_center = ants.get_center_of_mass(mri_mask)
    hist_center = ants.get_center_of_mass(hist_mask)

    shift = [f_c - m_c for f_c, m_c in zip(mri_center, hist_center)]

    new_origin: tuple[int, int] = tuple([o + s for o, s in zip(hist.origin, shift)])

    print("new origin:", new_origin)

    centered_hist = ants.image_clone(hist)
    centered_hist.set_origin(new_origin)

    centered_hist_mask = ants.image_clone(hist_mask)
    centered_hist_mask.set_origin(new_origin)

    out: dict[Literal["img", "mask"], ANTsImage] = {"img": centered_hist, "mask": centered_hist_mask}

    return out


def scale_and_align_to_ref(img: ANTsImage, reference: ANTsImage, interp: str = "linear") -> ANTsImage:
    """
    Calculate the spacing needed to match to the physical size of `reference`.
    i.e How big do `img` pixels need to be to match the phyiscal size of `reference`.

    new spacing = target physical space (i.e `reference` size in mm) / `img` resolution

    Also sets the direction and origin to that of `reference`.

    Args:
        img: The image to have the metadata calculated
        reference: The image from which to calculate the metadata from
        resample (optional): Whether to resample the image to the resolution of the reference. Defaults to False.
        interp (optional): The pixel interpolation mode for the resampling to use. Defaults to "linear".

    Returns:
        ANTsImage: The processed ants image
    """
    ref_phys_x = reference.shape[0] * reference.spacing[0]  # 176 * 0.1 = 17.6mm
    ref_phys_y = reference.shape[1] * reference.spacing[1]  # 176 * 0.1 = 17.6mm

    new_spacing_x = ref_phys_x / img.shape[0]
    new_spacing_y = ref_phys_y / img.shape[1]

    img.set_spacing((new_spacing_x, new_spacing_y))
    img.set_origin(reference.origin)
    img.set_direction(reference.direction)

    # if resample:
    #     return ants.resample_image_to_target(img, reference, interp_type=interp)  # type: ignore

    return img


def create_hist_volume(
    hist: ANTsImage,
    mri_key: str,
    dicom_params: DicomParams,
    out_path: Path,
    interp: str | None = "linear",
) -> ANTsImage:
    """
    Create a 3D nifti in the same shape as the dicom volume from which the MRI slides are from, but insert the slide/map at the
    appropriate Z-index, and the rest of the Z-slices are black. This allows for comparison/overlay with the original MRI volume.

    Args:
        hist: The histology image to insert into the volume. May be an actual histology slice or a computed map.
        Must be the same X and Y dimensions as the volume

        mri_key: The key of the MRI slide. The index of which the histology will be placed at.

        dicom_params: The main dicom parameters of the program, containing the volume and MRI slices.

        out_path: Filepath to output the nifti volume
    """
    mri_volume = dicom_params["volume"]

    # if hist.shape[0:2] != mri_volume.shape[0:2]:
    #     raise ValueError(f"Histology shape {hist.shape} doesn't match MRI shape {mri_volume.shape}")

    zeros: np.ndarray[tuple[int, int, int], np.dtypes.Float32DType] = np.zeros_like(mri_volume.numpy())  # type: ignore

    hist_scaled: ANTsImage = ants.resample_image_to_target(hist, mri_volume[:, :, 0], interp_type=interp)  # type: ignore
    print(f"{hist_scaled.shape=}")

    idx = dicom_params["slices"][mri_key]["index"]

    zeros[:, :, idx] = hist_scaled.numpy()

    out = ants.new_image_like(mri_volume, zeros)

    # Copy all metadata to the new volume, for completeness
    mri_sitk: sitk.Image = ants.to_sitk(mri_volume)
    out_sitk: sitk.Image = ants.to_sitk(out)
    for key in mri_sitk.GetMetaDataKeys():
        out_sitk.SetMetaData(key, mri_sitk.GetMetaData(key))

    out_metadata = ants.from_sitk(out_sitk)

    out_metadata.to_file(ensure_path_exists(out_path if is_file(out_path) else out_path / "volume.nii.gz"))
    return out_metadata


def transform_original_hist(
    hist_zero: ANTsImage,
    slice_details: HistSlicesDict,
    mri_zero: ANTsImage,
    mri_mask: ANTsImage,
    transform: list[str],
    out_path: Path,
) -> ANTsImage:
    """
    Apply the transform from the MRI-histology registration to the original high res histology image.
    We use a downscaled version during registration to improve registration quality, but we can then apply that transform
    to the original for visualisation purposes.
    """
    processed = prepare_hist(hist_zero, slice_details, mri_zero, mri_mask, center=False, resample=False)["img"]

    transformed: ANTsImage = ants.apply_transforms(
        transformlist=transform, fixed=processed, moving=processed, interpolator="linear"
    )  # type: ignore

    transformed.astype("uint8").to_file(ensure_path_exists(out_path / f"transformed.png"))

    return transformed


def prepare_mri(mri: ANTsImage) -> ThresholdDict:
    mri_bias_corrected = ants.abp_n4(mri)
    thresholded = threshold_img(mri_bias_corrected, destructive=True)
    print(f"{thresholded['mask'].dtype=}")
    # a = center_and_pad(thresholded, thresholded["img"].shape)
    (thresholded["mask"] * 255).astype("uint8").to_file("mri_mask.png")  # type: ignore
    return thresholded


def process_maps(
    slice_details: HistSlicesDict,
    transform: list[str],
    mri_processed: ANTsImage,
    out_path: Path,
    mri_key: str | None = "",
    dicom_params: DicomParams | None = None,
) -> dict[str, ProcessedMap]:
    if "maps" not in slice_details:
        raise AttributeError(f"Slice {slice_details['img'].path.name} has no associated maps.")

    ret_dict: dict[str, ProcessedMap] = {}

    for map_name, map_img in slice_details["maps"].items():
        map_processed = prepare_hist(
            map_img,
            slice_details,
            mri_processed,
            threshold=False,
            resample=True,
            interp="genericLabel",
            out_path=out_path / "maps" / "debug",
        )

        map_img.astype("uint8").to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_map_raw.png"))
        map_processed.astype("uint8").to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_map_processed.png"))

        map_transformed: ANTsImage = ants.apply_transforms(
            transformlist=transform,
            fixed=map_processed,
            moving=map_processed,
            interpolator="genericLabel",
        )  # type: ignore

        print(f"""
            {mri_processed.shape=}
            {mri_processed.spacing=}
            {mri_processed.origin=}
            {mri_processed.direction=}
            {map_transformed.shape=}
            {map_transformed.spacing=}
            {map_transformed.origin=}
            {map_transformed.direction=}
        """)

        map_transformed.astype("uint8").to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_map.png"))
        map_transformed.to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_map.tif"))

        if mri_key and dicom_params:
            # Create a volume with identical shape to the original MRI volume with the map inserted in the appropriate place
            create_hist_volume(
                map_transformed,
                mri_key,
                dicom_params,
                interp="genericLabel",
                out_path=out_path / "maps" / f"{map_name}.nii.gz",
            )

        mi_score: float = ants.image_mutual_information(mri_processed, map_transformed)
        print(mi_score)

        ret_dict[map_name] = {"img": map_transformed, "mutual_info": mi_score}

    return ret_dict


def allocate_hists(hists: list[HistSlicesDict], dicom_params: DicomParams) -> AllocatedHists[HistSlicesDict]:
    """Assign each hist slide an MRI slide to be coregistered with (assuming 5 hist slides and 2 MRI slides)"""

    ret_dict = defaultdict(list)
    valid_mri_keys = dicom_params["slices"].keys()

    def check_keys(key: str):
        if key in valid_mri_keys:
            ret_dict[key].append(hist)
        else:
            raise ValueError(f"{hist['img'].path.name} is set to register against non-existent MRI image '{mri_key}'.")

    for hist in hists:
        mri_key = hist["register_to"]

        if isinstance(mri_key, list):
            for key in mri_key:
                check_keys(key)
        else:
            check_keys(mri_key)

    return dict(ret_dict)


@overload
def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, dicom_params: DicomParams, return_reg_dict: Literal[True]
) -> AllocatedHists[RegistrationDict]:
    pass


@overload
def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, dicom_params: DicomParams, return_reg_dict: Literal[False]
) -> AllocatedHists[HistSlicesDict]:
    pass


def register_hist_within(
    hists: list[HistSlicesDict], hist_params: HistParams, dicom_params: DicomParams, return_reg_dict: bool = False
) -> AllocatedHists[T]:
    """
    Register hist slices against others. Returns a dict with hist slides allocated to MRI slides, optionally including the registration info. See `allocate_hists`
    """

    registered: AllocatedHists = defaultdict(list)

    fixed = hists[hist_params.get("fixed_image", 2)]["img"].greyscale_img()
    moving_dict = allocate_hists(hists, dicom_params)

    for mri_key, hist_dicts in moving_dict.items():
        for hist_dict in hist_dicts:

            moving = hist_dict["img"].greyscale_img()

            reg: RegistrationDict = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNRA")

            registered[mri_key].append(
                reg
                if return_reg_dict
                else {"img": LazyAntsImage(reg["warpedmovout"]), "rotation": hist_dict["rotation"]}
            )
            gc.collect()
    return dict(registered)


def threshold_img(img: LazyAntsImage | ANTsImage, *, destructive: bool) -> ThresholdDict:
    """
    Threshold an image into its largest single object. For MRI, this means removing the skull and keeping only the brain

    Args:
        img (LazyAntsImage | ANTsImage): The image to threshold
        destructive (bool, optional): Erode and dilate the image, produces worse results on histology but better for MRI. Not needed for subsequent uses after an initial destructive use.

    Raises:
        TypeError: If `img` is not of the correct type
        ValueError: `If the image is not 2D or 3D

    Returns:
        ThresholdDict: `img`: the thresholded image.

        `mask`: a binary mask (uint8) of the thresholded image.

        `region`: the properties of the largest region in the image. See `skimage.measure.regionprops`.
    """

    if isinstance(img, LazyAntsImage):
        img = img.img
    elif isinstance(img, ANTsImage):
        img = img
    else:
        raise TypeError("img must be a LazyAntsImage, or an AntsImage")

    np_arr = img.numpy()

    if np_arr.ndim == 2:
        footprint_fn = disk
    elif np_arr.ndim == 3:
        footprint_fn = ball
    else:
        raise ValueError("mri must be either 2D or 3D")

    thresh = threshold_otsu(np_arr)
    # binary = np_arr >= (thresh * 0.5)  # Manually fiddling with the threshold feels bad but it works?
    binary = np_arr >= thresh

    # Only erode and dilate if MRI image
    if destructive:
        binary = morphology.erosion(binary, footprint_fn(2))

    labeled = measure.label(binary)

    # Keep only the largest region in the binary image (the brain itself)
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda x: x.area)

    # Create the final mask using the largest region
    mask = np.zeros_like(np_arr, dtype=np.uint8)
    mask[labeled == largest.label] = 1

    if destructive:
        mask = morphology.dilation(mask, footprint_fn(4))
    # dilated_mask = solid_mask
    solid_mask: np.ndarray = ndimage.binary_fill_holes(mask)  # type: ignore

    final_region = measure.regionprops(solid_mask.astype(int))[0]

    brain = np_arr * solid_mask

    final_img = ants.new_image_like(img, brain)
    mask_ants = ants.new_image_like(img, solid_mask)

    return {"img": final_img, "mask": mask_ants, "region": final_region}


def create_checkerboard(
    img1: ANTsImage, img2: ANTsImage, out_path: Path, squares: tuple[int, int] | None = None
) -> sitk.Image:
    img = sitk.CheckerBoard(ants.to_sitk(img1), ants.to_sitk(img2), squares)
    ants.from_sitk(img).astype("uint8").to_file(ensure_path_exists(out_path / "checkerboard.png"))
    return img


def is_file(path: Path) -> bool:
    """If this path points to a file, regardless if that file exists or not"""
    return bool(path.suffix)


def ensure_path_exists(str_or_path: str | Path) -> str:
    """Mainly for use with ants filenames. Recursively create any necessary folders for the inputted filename."""
    path = Path(str_or_path)

    path = path.parent if is_file(path) else path  # If path is pointing to a file

    Path.mkdir(path, parents=True, exist_ok=True)
    return str(str_or_path)


def image_info(img: ANTsImage) -> ImageInfo:
    return LazyAntsImage(img).image_info


def best_rotation(hist: LazyAntsImage, mri: ANTsImage):
    out = {"img": None, "mi": float("-inf"), "rotation": 0}

    histimg = hist.greyscale_img()
    mri_sliced: ANTsImage = ants.slice_image(mri, axis=-1, idx=0).astype("float32")  # type: ignore

    mri_processed = prepare_mri(mri_sliced)["img"]

    hist_processed: ANTsImage = scale_and_align_to_ref(histimg, mri_processed, resample=True)  # type: ignore

    for deg in range(0, 360):
        print(deg)
        rotated = LazyAntsImage(hist_processed).rotate(deg)
        mi = ants.image_mutual_information(rotated, mri_processed) * -1  # MI is negative for some reason

        if mi > out["mi"]:
            out["img"] = rotated
            out["mi"] = mi
            out["rotation"] = deg

    out["mri"] = mri_processed
    return out


def generate_maps_for_params(hist_params: HistParams, scripts: dict[str, ScriptDict]) -> HistParams:
    """
    Run qupath scripts for each slice in the hist params

    Args:
        hist_params (HistParams): The hist params to run scripts for
        scripts (dict[str, str]): Key: the name of the resulting map (ie the key in the return value). Value: the name of the qupath script (see LazyAntsImage.run_qupath_script)

    Returns:
        HistParams: The updated hist params with the added maps
    """

    for slice_entry in hist_params["slices"]:

        for map_name, script_dict in scripts.items():
            map_output = slice_entry["img"].run_qupath_script(
                project_path=Path("23yqp"), script_name=script_dict["script_name"], script_args=script_dict["args"]
            )
            slice_entry["maps"][map_name] = map_output

    return hist_params


def plot_roi_intensity(
    hist_registered: ANTsImage,
    mri: ANTsImage,
    roi_hist: ROI,
    roi_mri: ROI,
    mi: float,
    out_path: Path,
):
    cropped_hist = ants.crop_indices(hist_registered, roi_hist[0], roi_hist[1]).numpy()
    cropped_mri = ants.crop_indices(mri, roi_mri[0], roi_mri[1]).numpy()

    flat_hist = np.ravel(cropped_hist)
    flat_mri = np.ravel(cropped_mri)

    normalised_hist = flat_hist / flat_hist.max()
    normalised_mri = flat_mri / flat_mri.max()

    if np.unique(normalised_hist)[0] == np.inf or np.unique(normalised_hist)[0] == np.inf:
        raise ValueError("Pixel normalisation divided by zero. Double check your ROI isn't all black")

    fig = plt.figure(layout="constrained")

    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:-1, 0:])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    mask = (normalised_hist > 0) & (normalised_mri > 0)
    masked_hist = normalised_hist[mask]
    masked_mri = normalised_mri[mask]

    ax1.scatter(masked_hist, masked_mri, color="blue", s=2)

    ax1.set(xlabel="Hist intensity", ylabel="MRI Intensity", title=f"Hist vs MRI intensity (MI: {mi*-1})")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.imshow(cropped_hist)
    ax2.set(title="Hist Image")

    ax3.imshow(cropped_mri)
    ax3.set(title="MRI Image")

    plt.savefig(out_path)
    # plt.show()


if __name__ == "__main__":
    dicom: DicomParams = {
        "volume": ants.dicom_read("23Y_SC2"),
        "slices": {
            "mri_1": {
                "img": LazyAntsImage(Path("23Y_SC2") / "MRIm08.dcm", dimension=3),
                "index": 7,
            },
            "mri_2": {
                "img": LazyAntsImage(Path("23Y_SC2") / "MRIm09.dcm", dimension=3),
                "index": 8,
            },
        },
    }
    hist_params: HistParams = {
        "loc_within": False,
        "fixed_image": 2,
        # "use_masks": True,
        "greyscale_type": "mean",
        "slices": [
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S1.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": ants.image_read(
                        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S1.svs_cellularity_100um.tif")
                    ),
                },
                "register_to": ["mri_1", "mri_2"],
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S2.svs"),
                "rotation": 110,
                "maps": {
                    # "cell count 80": ants.image_read(str(Path("23yqp") / "export" / "tile_cell_counts_80um.tif")),
                    "cell count 100": ants.image_read(
                        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S2.svs_cellularity_100um.tif")
                    ),
                },
                "register_to": ["mri_1", "mri_2"],
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S3.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": ants.image_read(
                        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S3.svs_cellularity_100um.tif")
                    ),
                },
                "register_to": ["mri_1", "mri_2"],
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S4.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": ants.image_read(
                        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S4.svs_cellularity_100um.tif")
                    ),
                },
                "register_to": ["mri_1", "mri_2"],
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S5.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": ants.image_read(
                        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S5.svs_cellularity_100um.tif")
                    ),
                },
                "register_to": ["mri_1", "mri_2"],
            },
        ],
    }

    reg: RegParams = {"type_of_transform": "SyNRA", "out_prefix": Path("1"), "use_initial_transform": True}
    # hist_params = generate_maps_for_params(hist_params, {"cell count": {"script_name": "cell count mask", "args": []}})
    ants_init(dicom_params=dicom, hist_params=hist_params, reg_params=reg)

    # mri = LazyAntsImage(Path("AV_GBM_PK_AV_GBM_PK_GCBA5.23Y_SC2__E4_P1") / "MRIm08.dcm", dimension=3).img

    # # mri.astype("uint8").to_file("mri.png")
    # # mri: ANTsImage = ants.slice_image(mri, axis=-1, idx=0)  # type: ignore
    # mri = ants.abp_n4(mri)
    # out = threshold_img(mri, destructive=True)
    # out["img"].astype("uint8").to_file("mrithresh.png")
    # (out["mask"] * 255).astype("uint8").to_file("mrimask.png")

    # hist = LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S5.svs").greyscale_img()
    # out = threshold_img(hist, destructive=False)
    # out["img"].astype("uint8").to_file("histthresh.png")
    # (out["mask"] * 255).astype("uint8").to_file("histmask.png")

    # from skimage import data

    # # Example IHC image
    # img_rgb = LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S5.svs").img

    # # Separate the stains from the IHC image
    # ihc_hed = rgb2hed(img_rgb.numpy())

    # # Create an RGB image for each of the stains
    # null = np.zeros_like(ihc_hed[:, :, 0])
    # ihc_h = ihc_hed[:, :, 0] #hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    # ihc_e = ihc_hed[:, :, 1] #hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    # ihc_d = ihc_hed[:, :, 2] #hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    # he_image = hed2rgb(np.stack((ihc_hed[:, :, 0], ihc_hed[:, :, 1], null), axis=-1))

    # print(he_image.shape, he_image.dtype)

    # gray_np = np.mean(he_image, axis=-1, dtype=np.float64)

    # inverted = gray_np.max() - gray_np
    # antsimg = ants.from_numpy(inverted, spacing=img_rgb.spacing, origin=img_rgb.origin, direction=img_rgb.direction, has_components=False)
    # (antsimg * 255).astype("uint8").to_file("he.png")

    # inverted = ihc_h.max() - ihc_h
    # h_image = ants.from_numpy(inverted, spacing=img_rgb.spacing, origin=img_rgb.origin, direction=np.eye(2), has_components=False)
    # (h_image * 255).astype("uint8").to_file("h.png")

    # inverted = ihc_e.max() - ihc_e
    # e_image = ants.from_numpy(inverted, spacing=img_rgb.spacing, origin=img_rgb.origin, direction=np.eye(2), has_components=False)
    # (e_image * 255).astype("uint8").to_file("e.png")

    # # # Display
    # # fig, axes = plt.subplots(3, 2, figsize=(7, 6), sharex=True, sharey=True)
    # # ax = axes.ravel()

    # # ax[0].imshow(img_rgb)
    # # ax[0].set_title("Original image")

    # # ax[1].imshow(ihc_h)
    # # ax[1].set_title("Hematoxylin")

    # # ax[2].imshow(ihc_e)
    # # ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    # # ax[3].imshow(ihc_d)
    # # ax[3].set_title("DAB")

    # # ax[4].imshow(he_image)
    # # ax[4].set_title("H&E")

    # # for a in ax.ravel():
    # #     a.axis("off")

    # # fig.tight_layout()
    # # plt.show()

    # mri = LazyAntsImage(Path("AV_GBM_PK_AV_GBM_PK_GCBA5.23Y_SC2__E4_P1") / "MRIm08.dcm", dimension=3)
    # print(mri.metadata)
