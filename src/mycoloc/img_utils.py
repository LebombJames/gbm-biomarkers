import ants
import matplotlib
import SimpleITK as sitk
import skimage.exposure as se
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import ball, disk

from src.mycoloc.__types import *
from src.mycoloc.config import DEBUG
from src.mycoloc.LazyAntsImage import LazyAntsImage
from src.mycoloc.utils import ensure_path_exists

matplotlib.use("agg")


def scale_and_align_to_ref(img: ANTsImage, reference: ANTsImage, interp: str = "nearestNeighbor") -> ANTsImage:
    """
    Calculate the spacing needed to match to the physical size of `reference`.
    i.e How big do `img` pixels need to be to match the phyiscal size of `reference`.

    new spacing = target physical space (i.e `reference` size in mm) / `img` resolution

    Also sets the direction and origin to that of `reference`.

    Args:
        img: The image to have the metadata calculated
        reference: The image from which to calculate the metadata from
        resample (optional): Whether to resample the image to the resolution of the reference. Defaults to False.
        interp (optional): The pixel interpolation mode for the resampling to use. Defaults to "nearestNeighbor".

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


def prepare_mri(mri: ANTsImage, out_path: Path | None = None) -> ThresholdDict:
    # print(np.unique(mri.numpy()))
    mri_bias_corrected = ants.abp_n4(mri, (0.01, 0.99, 256))

    # tissue_mask = mri_bias_corrected.numpy() > 0

    # p2, p98 = np.percentile(mri_bias_corrected.numpy()[tissue_mask], (2, 98))
    # stretched_image = se.rescale_intensity(mri_bias_corrected.numpy(), out_range=(mri.min(), np.iinfo(np.uint8).max))  # type: ignore

    # stretched_image[~tissue_mask] = 0

    # mri_bias_corrected = ants.new_image_like(mri_bias_corrected, stretched_image)

    thresholded = threshold_img(mri_bias_corrected, destructive=True)
    if DEBUG:
        # print(f"{thresholded['mask'].dtype=}")
        (thresholded["mask"] * 255).astype("uint8").to_file("mri_mask.png")  # type: ignore

    fig = Figure(layout="constrained")

    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle(f"MRI Processing")

    ax1.imshow(mri.view().T, cmap="gray")
    ax1.set(title="Before")
    ax1.set_axis_off()

    ax2.imshow(thresholded["img"].view().T, cmap="gray")
    ax2.set(title=f"After")
    ax2.set_axis_off()

    if out_path:
        fig.savefig(ensure_path_exists(out_path), dpi=300)

    thresholded["plot"] = fig

    return thresholded


def normalize_01(image: npt.NDArray):
    """
    Normalizes a numpy array to the range [0.0, 1.0].
    """
    img_float = image.astype(np.float64)

    img_min = img_float.min()
    img_max = img_float.max()

    if img_max == img_min:
        return np.zeros_like(img_float)

    normalized_img = (img_float - img_min) / (img_max - img_min)

    return normalized_img


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

    # print(np.unique(np_arr))
    # img.astype("uint8").to_file("wtf1.png")

    thresh = threshold_otsu(np_arr)
    # binary = np_arr >= (thresh * 0.5)  # Manually fiddling with the threshold feels bad but it works?
    binary = np_arr >= thresh

    # Only erode and dilate if MRI image
    if destructive:
        binary = morphology.erosion(binary, footprint_fn(3))

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


def compose_registration(reg_args: dict[str, Any], reg_types: list[str]) -> list[RegistrationDict]:
    reg_dicts: list[RegistrationDict] = []
    transformed_moving_mask = None

    for reg_type in reg_types:
        if reg_dicts:
            mov = reg_dicts[-1]["warpedmovout"]
        else:
            mov = reg_args["moving"]

        mov_mask = transformed_moving_mask or reg_args.get("moving_mask", None)
        reg: RegistrationDict = ants.registration(
            **reg_args, moving=mov, moving_mask=mov_mask, type_of_transform=reg_type
        )

        if "moving_mask" in reg_args:
            transformed_moving_mask = ants.apply_transforms(
                transformlist=reg["fwdtransforms"],
                fixed=mov_mask,
                moving=mov_mask,
                interpolator="genericLabel",
            )

        reg_dicts.append(reg)

    return reg_dicts  # Take all the fwdtransforms and apply them, the image is probably very warped at this point


def correct_map_interp(original_map: ANTsImage, resampled_map: ANTsImage) -> ANTsImage:
    r"""
    When we transform the map, we may be destroying or creating pixels, so we need to correct cell counts for that.
    ```
    OLD:
    ---------
    | 10 | 10 |
    |----|----|
    | 10 | 10 |
    ---------
        |
        | (linear)
        |
        \/
    NEW:
    ---------
    |         |
    |    10   | WRONG!
    |         |
    ---------
    npixels(old) = 4
    npixels(new) = 1
    scale factor = 4/1 = 4
    10 * 4 = 40 CORRECT!
    ```
    """
    orig_total_pixels: int = original_map.shape[0] * original_map.shape[1]
    new_total_pixels: int = resampled_map.shape[0] * resampled_map.shape[1]

    scale_factor: float = orig_total_pixels / new_total_pixels

    return ants.new_image_like(resampled_map, resampled_map.view() * scale_factor)


def create_checkerboard(
    mri: ANTsImage,
    hist: ANTsImage,
    out_path: Path,
    mri_mask: ANTsImage | None = None,
    hist_mask: ANTsImage | None = None,
    squares: tuple[int, int] | None = None,
    intensity_rescale: float | None = 0.5,
) -> ANTsImage:

    # if mri_mask:
    #     normalized = se.equalize_hist(mri.numpy(), mask=mri_mask.numpy())

    mri_np = mri.numpy()
    tissue_mask = mri_np > 0

    p2, p98 = np.percentile(mri_np[tissue_mask], (2, 98))
    stretched_image = se.rescale_intensity(
        mri_np, out_range=(hist.min(), hist.max() * 1)  # type: ignore
    )

    stretched_image[~tissue_mask] = 0

    mri = ants.new_image_like(mri, stretched_image)
    # if mri_mask and hist_mask:
    #     mri_mask = ants.resample_image_to_target(mri_mask, hist_mask)  # type: ignore
    #     mri_mask = ants.copy_image_info(hist_mask, mri_mask)

    # mri = ants.histogram_match_image2(mri, hist, source_mask=mri_mask, reference_mask=hist_mask)

    # mri = ants.histogram_equalize_image(mri)

    mri = ants.resample_image_to_target(mri, hist, interp_type="nearestNeighbor")  # type: ignore
    mri = ants.copy_image_info(hist, mri)

    mri_s = ants.to_sitk(mri)
    hist_s = ants.to_sitk(hist)

    img = sitk.CheckerBoard(mri_s, hist_s, squares)
    ants_img = ants.from_sitk(img)

    ants_img.to_file(ensure_path_exists(out_path / "checkerboard.nii.gz"))
    ants_img.astype("uint8").to_file(ensure_path_exists(out_path / "checkerboard.png"))
    return ants_img
