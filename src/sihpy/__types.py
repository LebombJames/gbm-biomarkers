from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, TypeVar
from dataclasses import dataclass, field
from typing_extensions import NotRequired
from pathlib import Path

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from ants import ANTsImage
    from matplotlib.figure import Figure
    from skimage.measure._regionprops import RegionProperties
    from src.sihpy.LazyAntsImage import LazyAntsImage


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
    dimensions: tuple[int | float, ...]
    spacing: tuple[int | float, ...]
    origin: tuple[int | float, ...]
    direction: npt.NDArray[np.float32]


class ImageInfo(TypedDict):
    shape: tuple[int, ...]
    physical_shape: tuple[int | float, ...]
    spacing: tuple[int | float, ...]
    origin: tuple[int | float, ...]
    direction: np.ndarray[tuple[int, int], np.dtype[Any]]


class DicomParams(TypedDict):
    slices: dict[str, MRISliceDict]
    volume: ANTsImage


@dataclass
class DicomParamsDC:
    slices: dict[str, MRISliceDict]
    volume: ANTsImage


class MRISliceDict(TypedDict):
    img: LazyAntsImage
    index: int
    """
    Index within the volume that this slice occupies, starting from 0 (so if it's `MRIm08.dcm`, the `index` is `7`)
    """


GreyscaleModes = Literal["mean", "red", "green", "blue", "h", "e", "h&e"] | set[Literal["red", "green", "blue"]]


class HistParams(TypedDict):
    slices: list[HistSlicesDict]
    loc_within: bool
    """Localise the slices against slide 3 before colocalising against the MRI"""
    fixed_image: int
    """Array index of the fixed image if localising within hist slides. `2` (Slide 3) by default"""
    # use_masks: bool
    greyscale_type: GreyscaleModes
    split_multiple_register_to: bool
    """If true, histology slices with an array of MRI keys to register to will have their pixel intensity split amongst them. By default, this is 1/n, where n is the length of `register_to`, but `middle_slice_factor` can customise this."""


@dataclass
class HistParamsDC:
    slices: list[HistSlicesDict] = field(default_factory=list)
    loc_within: bool = False
    """Localise the slices against slide 3 before colocalising against the MRI"""
    fixed_image: int = 2
    """Array index of the fixed image if localising within hist slides. `2` (Slide 3) by default"""
    greyscale_type: GreyscaleModes = "mean"
    """How to reduce an RGB image to a single channel.

    `"mean"`: average RGB channels together.

    Set including `"red"`, `"green"` or `"blue"`: Average specified channels together. All 3 is equivalent to `"mean"`. A single channel is equivalent to passing the string alone.

    `"red"`, `"green"`, or `"blue"`: Use only the specified channel

    `"h"`: Haematoxylin OD

    `"e"`: Eosin OD

    `"h&e"`: H&E OD

    """
    split_multiple_register_to: bool = True
    """If true, histology slices with an array of MRI keys to register to will have their pixel intensity split amongst them. By default, this is 1/n, where n is the length of `register_to`, but `middle_slice_factor` can customise this."""


class RegParams(TypedDict):
    type_of_transform: str
    out_prefix: Path
    use_initial_transform: bool


@dataclass
class RegParamsDC:
    type_of_transform: str = "SyN"
    out_prefix: Path = Path("out")
    use_initial_transform: bool = False


class AnimalParams(TypedDict):
    mri: DicomParams
    hist: HistParams


class HistSlicesMaps(TypedDict):
    map_img: LazyAntsImage
    """The image of the map"""
    necrosis_correct: bool
    """
    Whether to apply Tumour% necrosis correction (use only for cell density maps)
    """
    combine_type: Literal["add", "mean"]


@dataclass
class HistSlicesMapsDC:
    map_img: LazyAntsImage
    """The image of the map"""
    necrosis_correct: bool = False
    """
    Whether to apply Tumour% necrosis correction (use only for cell density maps)
    """
    combine_type: Literal["add", "mean"] = "add"


class HistSlicesDict(TypedDict):
    img: LazyAntsImage
    rotation: int
    """A clockwise rotation in degrees to apply to the image.

    Note: Rotation is applied *after* cropping.
    """
    maps: dict[str, HistSlicesMaps]
    """Maps computed using `img`, which will be registered using the same transform calculated on `img` from the registration."""
    crop: NotRequired[ROI]
    """
    The indicies to crop the image with. `((X1, Y1), (X2, Y2))`, where `X1` and `Y1` are
    the minimum indicies to crop at, and `X2` and `Y2` are the maximum to crop at.

    Note: Cropping is applied *before* rotation.
    """
    register_to: str | list[str]
    "The key of the MRI slide to register this histology to. If a list of strings, the histology will be allocated to all corresponding MRI slides. See `DicomParams`."
    middle_slice_factor: NotRequired[dict[str, float]]
    """If `register_to` is a list, a dict of floats (0,1] of weightings to apply for the slice at the MRI slice corresponding to the key. E.g {"mri_1": 0.5, "mri_2": 0.5} will halve the slice's intensities across its two MRI slices."""
    necrosis_map: LazyAntsImage | None
    "The necrosis map used to correct the map images in `maps`"


@dataclass
class HistSlicesDictDC:
    img: LazyAntsImage
    crop: ROI
    """
    The indicies to crop the image with. `((X1, Y1), (X2, Y2))`, where `X1` and `Y1` are
    the minimum indicies to crop at, and `X2` and `Y2` are the maximum to crop at.

    Note: Cropping is applied *before* rotation.
    """
    middle_slice_factor: dict[str, float]  # = {"mri_1": 0.5, "mri_2": 0.5}
    """If `register_to` is a list, a dict of floats (0,1] of weightings to apply for the slice at the MRI slice corresponding to the key. E.g {"mri_1": 0.5, "mri_2": 0.5} will halve the slice's intensities across its two MRI slices."""
    rotation: int = 0
    """A clockwise rotation in degrees to apply to the image.

    Note: Rotation is applied *after* cropping.
    """
    maps: dict[str, HistSlicesMaps] = field(default_factory=dict)
    """Maps computed using `img`, which will be registered using the same transform calculated on `img` from the registration."""
    register_to: str | list[str] = ""
    "The key of the MRI slide to register this histology to. If a list of strings, the histology will be allocated to all corresponding MRI slides. See `DicomParams`."
    necrosis_map: LazyAntsImage | None = None
    "The necrosis map used to correct the map images in `maps`"


T = TypeVar("T", "LazyAntsImage", RegistrationDict, "ANTsImage", HistSlicesDict)
AllocatedHists = dict[str, list[T]]
"""Group hist slides based on which MRI slice they should be registered to. The values may be either an Ants Image of the moving image registered, or the dict returned from the registration, which allows for the transformation function to be accessed."""


class ScriptDict(TypedDict):
    script_name: str
    args: list[str]


ROI = tuple[tuple[int, int], tuple[int, int]]


class RegPlots(TypedDict):
    name: str
    checkerboard: list[RegPlot]
    map_overview: list[RegPlot]
    transformed_original: list[RegPlot]
    mri_overview: list[RegPlot]


class RegPlot(TypedDict):
    img: ANTsImage | dict[str, ANTsImage] | dict[str, Any] | Figure
    mri_key: str
    hist_name: str | None
    animal_name: NotRequired[str]
    map_name: NotRequired[str]


class ThresholdDict(TypedDict):
    img: ANTsImage
    mask: ANTsImage
    region: RegionProperties
    plot: NotRequired[Figure]


class ProcessedMap(TypedDict):
    img: LazyAntsImage
    mutual_info: float
    mri_key: str
    map_name: str
    combine_type: Literal["add", "mean"]
    control_mi: float


class GridDims(NamedTuple):
    nrows: int
    ncols: int
