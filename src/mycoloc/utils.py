from __future__ import annotations

import pprint
from pathlib import Path

from ants import ANTsImage
from tqdm import tqdm

from src.mycoloc.__types import *
from src.mycoloc.LazyAntsImage import LazyAntsImage


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


class ANTsPrettyPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if type(object).__name__ == "ANTsImage":
            # Return a more readable string for Ants images (plus readable/recursive flags to match original return value)
            return "<ANTsImage>", True, False

        return super().format(object, context, maxlevels, level)


def mypprint(x):
    ANTsPrettyPrinter(width=200, indent=0).pprint(x)


global progress
progress = tqdm()
