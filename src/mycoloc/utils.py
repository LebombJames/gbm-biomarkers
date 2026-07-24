from __future__ import annotations

import asyncio
import math
import pprint
import re
import time
from functools import wraps
from itertools import starmap
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from src.mycoloc.__types import *


def n_subplots(n: int, orientation: Literal["wide", "long"] = "wide") -> GridDims:
    if orientation == "long":
        nrows = math.ceil(math.sqrt(n))
        ncols = math.ceil(n / nrows)
    else:
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

    return GridDims(nrows=nrows, ncols=ncols)


def pretty_mri_key(key: str) -> str:
    return key.replace("_", " ").upper()


def animal_id_from_filename(filename: Path | str) -> str:
    if not filename:
        return ""

    animal = re.search(r"23\w_", str(filename))
    if animal:
        animal = animal.group()
    else:
        raise ValueError(f"No animal found! {filename}")

    return animal.upper()[:-1]


def slice_number_from_filename(filename: Path | str) -> str:
    slide = re.search("S\\d", str(filename))
    if slide:
        slide = slide.group()
    else:
        raise ValueError(f"No slide found! {filename}")
    return slide


def pretty_hist_filename(filename: Path | str) -> str:
    if not filename:
        return ""

    filename = str(filename)

    animal = animal_id_from_filename(filename)

    slide = slice_number_from_filename(filename)

    return f"{animal}-{slide[1:]}"


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


def multi_index_to_str(midx: pd.Index, sep="_"):
    fstr = sep.join(["{}"] * midx.nlevels)
    return pd.Index(starmap(fstr.format, midx))


class ANTsPrettyPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if type(object).__name__ == "ANTsImage":
            # Return a more readable string for Ants images
            # (repr_string, isreadable, isrecursive)
            return "<ANTsImage>", True, False

        return super().format(object, context, maxlevels, level)


def mypprint(x):
    ANTsPrettyPrinter(width=200, indent=0).pprint(x)


progress = tqdm(total=0)
