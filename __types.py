from __future__ import annotations
from typing import TypedDict, Literal, TypeVar, Any
from typing_extensions import NotRequired

from ants import ANTsImage
import numpy as np
import numpy.typing as npt
from coloc import LazyAntsImage
from pathlib import Path
from skimage.measure._regionprops import RegionProperties


class RegistrationDict(TypedDict):
    warpedmovout: ANTsImage
    """Moving image warped to space of fixed image."""
    warpedfixout: ANTsImage
    """Fixed image warped to space of moving image."""
    fwdtransforms: list[str]
    """Transforms to move from moving to fixed image."""
    invtransforms: list[str]
    """Transforms to move from fixed to moving image"""


class AntsHeader(TypedDict):
    pixelclass: str
    pixeltype: str
    nDimensions: int
    nComponents: int
    dimensions: tuple[float, float, float]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: npt.NDArray[np.float64]


class DicomParams(TypedDict):
    slices: dict[str, MRISliceDict]
    volume: ANTsImage


class MRISliceDict(TypedDict):
    img: LazyAntsImage
    index: int
    """
    Index within the volume that this slice occupies, starting from 0 (so if it's `MRIm08.dcm`, the `index` is `7`)
    """


class HistParams(TypedDict):
    slices: list[HistSlicesDict]
    loc_within: bool
    """Localise the slices against slide 3 before colocalising against the MRI"""
    fixed_image: int
    """Array index of the fixed image if localising within hist slides. `2` (Slide 3) by default"""
    # use_masks: bool
    # """
    # Threshold slides before registration. A `mask` value can be provided in each slice, otherwise a mask will be calculated using ants.
    # """
    greyscale_type: GreyscaleModes


GreyscaleModes = Literal["mean", "red", "green", "blue", "h", "e", "h&e"] | set[Literal["red", "green", "blue"]]


class RegParams(TypedDict):
    type_of_transform: str
    out_prefix: Path
    use_initial_transform: bool


class HistSlicesMaps(TypedDict):
    map_img: ANTsImage
    """The image of the map"""
    necrosis_correct: bool
    """
    False: No correction

    1: Cellularity necrosis correction

    2: Tumour% necrosis correction
    """
    combine_type: Literal["add", "mean"]


class HistSlicesDict(TypedDict):
    img: LazyAntsImage
    rotation: int
    """A clockwise rotation in degrees to apply to the image.

    Note: Rotation is applied *after* cropping.
    """
    maps: dict[str, HistSlicesMaps]
    """Maps computed using `img`, which will be registered using the same transform calculated on `img` from the registration."""
    crop: NotRequired[tuple[tuple[int, int], tuple[int, int]]]
    """
    The indicies to crop the image with. `((X1, Y1), (X2, Y2))`, where `X1` and `Y1` are
    the minimum indicies to crop at, and `X2` and `Y2` are the maximum to crop at.

    Note: Cropping is applied *before* rotation.
    """
    register_to: str | list[str]
    "The key of the MRI slide to register this histology to. If a list of strings, the histology will be allocated to all corresponding MRI slides. See `DicomParams`."
    necrosis_map: ANTsImage | None
    "The necrosis map used to correct the map images in `maps`"


T = TypeVar("T", LazyAntsImage, RegistrationDict, ANTsImage, HistSlicesDict)
AllocatedHists = dict[str, list[T]]
"""Group hist slides based on which MRI slice they should be registered to. The values may be either an Ants Image of the moving image registered, or the dict returned from the registration, which allows for the transformation function to be accessed."""

HistSlices = list[LazyAntsImage]


class ScriptDict(TypedDict):
    script_name: str
    args: list[str]


ROI = tuple[tuple[int, int], tuple[int, int]]


class ThresholdDict(TypedDict):
    img: ANTsImage
    mask: ANTsImage
    region: RegionProperties


class ImageInfo(TypedDict):
    shape: tuple[int, ...]
    physical_shape: tuple[int | float, ...]
    spacing: tuple[int | float, ...]
    origin: tuple[int | float, ...]
    direction: np.ndarray[tuple[int, int], np.dtype[Any]]


class ProcessedMap(TypedDict):
    img: ANTsImage
    mutual_info: float
    mri_key: str
    map_name: str
    combine_type: Literal["add", "mean"]
