from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING, Literal, TypeVar

import ants
import numpy as np
import numpy.typing as npt
from pathlib import Path
from coloc import LazyAntsImage


class RegistrationDict(TypedDict):
    warpedmovout: ants.ANTsImage
    """Moving image warped to space of fixed image."""
    warpedfixout: ants.ANTsImage
    """Fixed image warped to space of moving image."""
    fwdtransforms: ants.ANTsTransform
    """Transforms to move from moving to fixed image."""
    invtransforms: ants.ANTsTransform
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
    slices: tuple[LazyAntsImage, LazyAntsImage]


class HistParams(TypedDict):
    slices: tuple[LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage]
    loc_within: bool
    """Localise the slices against slide 3 before colocalising against the MRI"""
    fixed_image: int
    """Array index of the fixed image if localising within hist slides. Slide 3 by default"""
    slide_3: Literal[0] | Literal[1] | Literal[2] | Literal[3]
    """
    0: Include slide 3 in no colocalisations (slide 1 and 2 against MRI slide 1, 4 and 5 against MRI slide 2)

    1: Colocalise slide 1, 2, and 3 against MRI slide 1, and 3, 4, and 5 against MRI slide 2

    2: Colocalise slide 1 and 2 against MRI slide 1, and 3, 4, 5 against MRI slide 2

    3: Include slide 3 in both colocalisations (slide 1, 2, and 3 against MRI slide 1, and 3, 4, and 5 against MRI slide 2)
    """
    downsample_svs: int
    """SVS files are massive, so we should downsample them."""


T = TypeVar("T", LazyAntsImage, RegistrationDict, ants.ANTsImage)
AllocatedHists = dict[Literal["mri_1"] | Literal["mri_2"], list[T]]

HistSlices = tuple[LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage, LazyAntsImage]
