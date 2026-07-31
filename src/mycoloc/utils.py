from __future__ import annotations

import asyncio
import math
import pprint
import re
import time
from functools import wraps
from itertools import starmap
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from tqdm import tqdm

from src.mycoloc.__types import GridDims


def create_subplot_grid(n: int, orientation: Literal["wide", "long"] = "wide") -> GridDims:
    """
    Matplotlib utility funciton to create a grid of subplots for a given number of subplots as close to a square as possible

    Args:
        n (int): Number of subplots
        orientation ("wide" | "long", optional): If not perfectly square, should the output be long or wide (e.g 2x3 or 3x2). Defaults to "wide".

    Returns:
        GridDims: NamedTuple with `nrows` and `ncols` properties
    """
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
    """
    Decorator to print time taken to run a function. Supports both sync and async functions.

    Args:
        func (Callable): Function to time
    """
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
    """
    Convert the levels of a Pandas Multi-index to strings

    Args:
        midx (pd.Index): The multi-index
        sep (str, optional): The character to join the levels of the multi-index. Defaults to "_".

    Returns:
        pd.Index: The index, where each level is now a string comprised of the levels of the Multi-index
    """
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


progress = tqdm(total=0, delay=5)
