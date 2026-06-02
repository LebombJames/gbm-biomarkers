from __future__ import annotations
from typing import TypedDict, Literal, TypeVar, Any
from typing_extensions import NotRequired

import ants
from ants import ANTsImage
import numpy as np
import numpy.typing as npt
from coloc import LazyAntsImage
from pathlib import Path


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
    slices: dict[Literal["mri_1", "mri_2"], LazyAntsImage]


class HistParams(TypedDict):
    slices: tuple[HistSlicesDict, HistSlicesDict, HistSlicesDict, HistSlicesDict, HistSlicesDict]
    loc_within: bool
    """Localise the slices against slide 3 before colocalising against the MRI"""
    fixed_image: int
    """Array index of the fixed image if localising within hist slides. Slide 3 by default"""
    slide_3_mode: Literal[0, 1, 2, 3]
    """
    0: Include slide 3 in no colocalisations (slide 1 and 2 against MRI slide 1, 4 and 5 against MRI slide 2)

    1: Colocalise slide 1, 2, and 3 against MRI slide 1, and 3, 4, and 5 against MRI slide 2

    2: Colocalise slide 1 and 2 against MRI slide 1, and 3, 4, 5 against MRI slide 2

    3: Include slide 3 in both colocalisations (slide 1, 2, and 3 against MRI slide 1, and 3, 4, and 5 against MRI slide 2)
    """


class RegParams(TypedDict):
    type_of_transform: str
    out_prefix: Path


class HistSlicesDict(TypedDict):
    img: LazyAntsImage
    rotation: int
    """A clockwise rotation in degrees to apply to the image.

    Note: Rotation is applied *after* rotation.
    """
    maps: NotRequired[dict[str, ANTsImage]]
    """Maps computed using `img`, which will be registeted using the same transform calculated on `img`."""
    crop: NotRequired[tuple[tuple[int, int], tuple[int, int]]]
    """
    The indicies to crop the image with. `((X1, Y1), (X2, Y2))`, where `X1` and `Y1` are
    the minimum indicies to crop at, and `X2` and `Y2` are the maximum to crop at.

    Note: Cropping is applied *before* rotation.
    """


T = TypeVar("T", LazyAntsImage, RegistrationDict, ANTsImage, HistSlicesDict)
AllocatedHists = dict[Literal["mri_1", "mri_2"], list[T]]
"""Group hist slides based on which MRI slice they should be registered to. The values may be either an Ants Image of the moving image registered, or the dict returned from the registration, which allows for the transformation function to be accessed."""

HistSlices = tuple[LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage]


class ImageInfo(TypedDict):
    shape: tuple[int, ...]
    physical_shape: tuple[int | float, ...]
    spacing: tuple[int | float, ...]
    origin: tuple[int | float, ...]
    direction: np.ndarray[tuple[int, int], np.dtype[Any]]
