import ants
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import ball, disk

from src.mycoloc.__types import *
from src.mycoloc.config import DEBUG
from src.mycoloc.utils import ensure_path_exists
from src.mycoloc.LazyAntsImage import LazyAntsImage


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


def prepare_mri(mri: ANTsImage) -> ThresholdDict:
    mri_bias_corrected = ants.abp_n4(mri)
    thresholded = threshold_img(mri_bias_corrected, destructive=True)
    if DEBUG:
        print(f"{thresholded['mask'].dtype=}")
        (thresholded["mask"] * 255).astype("uint8").to_file("mri_mask.png")  # type: ignore
    return thresholded


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
