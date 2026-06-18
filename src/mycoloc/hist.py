import gc
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal, overload

import ants
import SimpleITK as sitk
from ants import ANTsImage

from src.mycoloc.__types import *
from src.mycoloc.img_utils import align_centers_physically, scale_and_align_to_ref, threshold_img
from src.mycoloc.utils import ensure_path_exists, is_file, mypprint
from src.mycoloc.LazyAntsImage import LazyAntsImage


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


def create_hist_volume(
    hist_dict: dict[str, ANTsImage],
    dicom_params: DicomParams,
    out_path: Path,
    interp: str | None = "linear",
) -> ANTsImage:
    """
    Create a 3D nifti in the same shape as the dicom volume from which the MRI slides are from, but insert the slide/map at the
    appropriate Z-index, and the rest of the Z-slices are black. This allows for comparison/overlay with the original MRI volume.

    Args:
        hist_dict: A dictionary, where the keys are the MRI keys, and the values are the histology images. The images will be inserted into the volume at the index of the MRI slides associated with the keys (see dicom_params) E.g `{"mri_1": hist1, "mri_2": hist2}`

        dicom_params: The main dicom parameters of the program, containing the volume and MRI slices.

        out_path: Filepath to output the nifti volume

        interp: if the hist images aren't the same shape as the MRI slides, the interpolation method to apply during resampling
    """
    mri_volume = dicom_params["volume"]

    # if hist.shape[0:2] != mri_volume.shape[0:2]:
    #     raise ValueError(f"Histology shape {hist.shape} doesn't match MRI shape {mri_volume.shape}")

    # mypprint(hist_dict)

    zeros = np.zeros_like(mri_volume.numpy())

    mapped = {dicom_params["slices"][mri_key]["index"]: img for mri_key, img in hist_dict.items()}

    # mypprint(mapped)

    for idx, img in mapped.items():
        if img.shape[0:2] != mri_volume.shape[0:2]:
            hist_scaled: ANTsImage = ants.resample_image_to_target(img, mri_volume[:, :, 0], interp_type=interp)  # type: ignore
        else:
            hist_scaled = img

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
