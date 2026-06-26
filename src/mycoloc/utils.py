from __future__ import annotations

import asyncio
import pprint
import time
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

import ants
import numpy as np
from ants import ANTsImage
from tqdm import tqdm

from src.mycoloc.__types import *
from src.mycoloc.LazyAntsImage import LazyAntsImage

E = TypeVar("E")


def find(arr: list[E], el: E) -> E | None:
    try:
        return arr[arr.index(el)]
    except ValueError:
        return None


def func_timer(func: Callable):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_inner(*args, **kwargs):
            start = time.perf_counter()
            ret = await func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func.__name__} completed in {end - start:.4f}s")
            return ret

        return async_inner

    else:

        @wraps(func)
        def sync_inner(*args, **kwargs):
            start = time.perf_counter()
            ret = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func.__name__} completed in {end - start:.4f}s")
            return ret

        return sync_inner


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


def new_image_like_any_shape(image: ANTsImage, data: np.ndarray) -> ANTsImage:
    return ants.from_numpy(
        data=data,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        has_components=image.has_components,
        is_rgb=image.is_rgb,
    )


class ANTsPrettyPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if type(object).__name__ == "ANTsImage":
            # Return a more readable string for Ants images (plus readable/recursive flags to match original return value)
            return "<ANTsImage>", True, False

        return super().format(object, context, maxlevels, level)


def mypprint(x):
    ANTsPrettyPrinter(width=200, indent=0).pprint(x)


global progress
progress = tqdm(total=0)
