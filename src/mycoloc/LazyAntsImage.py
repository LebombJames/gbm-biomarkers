from __future__ import annotations

import math
import os
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast, overload

import ants
import tiffslide
from ants import ANTsImage
from PIL import Image
from pydicom import datadict
from skimage.color import hed2rgb, rgb2hed

from src.mycoloc.__types import *


class LazyAntsImage:
    """
    A wrapper around Path that lazy loads an AntsImage when called, or returns a cached version.
    """
    def __new__(cls, path_or_img, *args, **kwargs):
        # Immediately return the existing param if its a LazyAntsImage
        if isinstance(path_or_img, LazyAntsImage):
            return path_or_img

        return super().__new__(cls)

    def __init__(
        self,
        path_or_img: Path | ANTsImage | LazyAntsImage,
        level: int = 2,
        *args,
        **kwargs,
    ) -> None:
        self.level = level
        self.args = args
        self.kwargs = kwargs

        self.maps: dict[str, LazyAntsImage] = {}

        from src.mycoloc.utils import progress
        self.progress = progress

        if isinstance(path_or_img, LazyAntsImage):
            return  # Handled in __new__
        elif isinstance(path_or_img, Path):
            self.path = path_or_img
        elif isinstance(path_or_img, ANTsImage):
            self.img = path_or_img
        else:
            raise TypeError("Neither a Path or an AntsImage were provided.")

    @cached_property
    def img(self) -> ANTsImage:
        """Return the cached Ants image, or load it from the path if not loaded yet."""

        if self.path is None:
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately before
            # this whole function is even called.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        self.progress.write(f"Loading image: {self.path}")

        return self.svs_read(self.path) if self.is_hist else ants.image_read(str(self.path), *self.args, **self.kwargs)

    @property
    def is_hist(self):
        return self.path.suffix == ".svs"

    @property
    def header_info(self) -> AntsHeader:
        if self.path is None and self.img is not None:
            raise ValueError("Header Info is not available for in memory images.")

        return ants.image_header_info(str(self.path))

    @property
    def metadata(self):
        if self.path is None and self.img is not None:
            raise ValueError("Metadata Info is not available for in memory images.")

        metadata: dict[str, Any] = ants.read_image_metadata(str(self.path))

        translated: dict[str, Any] = {}
        for key, value in metadata.items():
            # Check if the key matches the "XXXX|YYYY" DICOM tag pattern
            split = key.split("|")
            if len(split) != 2:
                # Include any other keys that don't fit the pattern as a fallback
                translated[key] = value
                continue

            try:
                # Pydicom can take a tuple of the key as (XXXX, YYYY)
                readable_name = datadict.keyword_for_tag(split)  # type: ignore

                new_key = readable_name if readable_name else key
                translated[new_key] = value

            except:
                # Fallback just in case the lookup fails
                translated[key] = value

        return translated

    def greyscale_img(self, mode: GreyscaleModes = "mean") -> ANTsImage:
        """
        Returns a greyscale image constructed from the means of the RGB channels, and inverts it (255 becomes 0)

        Args:
            mode: How the greyscale is calculated. Accepts "red", "green", "blue", to use only those channels, "mean" to average all three, or a set containing any combination of "red", "green", and "blue" to average those. Defaults to "mean".
        """
        img = self.img

        if img.components <= 1:
            return img

        # In an AntsImage, the 0th colour channel is red, the 1st green, and 2nd blue.
        colour_map: dict[Literal["red", "green", "blue"], Literal[0, 1, 2]] = {"red": 0, "green": 1, "blue": 2}

        img_np: npt.NDArray[np.float32] = img.numpy()

        if mode == "mean" or mode == set(colour_map.keys()):
            # The mean of RGB values is grayscale
            gray_np = np.mean(img_np, axis=-1, dtype=np.float64)

            # Invert. Ants seems to work best with a black background
            inverted = gray_np.max() - gray_np
        elif mode in ["h", "e", "h&e"]:
            mode = cast(Literal["h", "e", "h&e"], mode)
            ihc_hed = rgb2hed(img.numpy())

            null = np.zeros_like(ihc_hed[:, :, 0])
            ihc_h = ihc_hed[:, :, 0]  # hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            ihc_e = ihc_hed[:, :, 1]  # hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            ihc_d = ihc_hed[:, :, 2]

            if mode == "h&e":
                he_image = hed2rgb(np.stack((ihc_hed[:, :, 0], ihc_hed[:, :, 1], null), axis=-1))
                gray_np = np.mean(he_image, axis=-1, dtype=np.float64)

                inverted = gray_np.max() - gray_np
            elif mode == "e":
                inverted = ihc_e.max() - ihc_e
            elif mode == "h":
                inverted = ihc_h.max() - ihc_h
        elif isinstance(mode, str):  # "red", "green", "blue"
            try:
                colour_idx = colour_map[cast(Literal["red", "green", "blue"], mode)]
            except KeyError as e:
                raise KeyError(f"{mode} is not a valid greyscale mode") from e

            colour_np = ants.split_channels(img)[colour_idx].numpy()
            inverted = colour_np.max() - colour_np

        elif isinstance(mode, set):
            try:
                channels_idx = {colour_map[colour] for colour in mode}
            except KeyError as e:
                raise KeyError(f"{mode} is not a valid greyscale mode") from e

            split_channels = ants.split_channels(img)
            channels = [split_channels[channel_idx] for channel_idx in channels_idx]
            merged = ants.merge_channels(channels).numpy()

            merged_np = np.mean(merged, axis=-1, dtype=np.float64)

            inverted = merged_np.max() - merged_np

        return ants.from_numpy(inverted, origin=img.origin, spacing=img.spacing, direction=img.direction)

    @property
    def image_info(self) -> ImageInfo:
        return {
            "shape": self.img.shape,
            "physical_shape": self.img.physical_shape,
            "spacing": self.img.spacing,
            "origin": self.img.origin,
            "direction": self.img.direction,
        }

    def run_qupath_script(
        self, project_path: Path, script_name: str, out_filename: str = "", script_args: list[str] | None = None
    ) -> LazyAntsImage:
        """
        Run a qupath script that takes this image as an argument (`args[0]` in the groovy script),
        and creates an output image, which we read and return
        """
        if self.path is None:
            raise ValueError("Scripts not available for in memory images.")

        if not script_args:
            script_args = []

        processed_args = []
        for arg in script_args:
            processed_args.append("--args")
            processed_args.append(arg)

        home = Path.home()
        quPath_dir = home / "AppData" / "Local" / "QuPath-0.7.0"

        cwd = Path(os.getcwd())
        out_path = cwd / f"{out_filename or self.path.name}-{script_name}.tif"
        project = cwd / project_path

        args = [
            quPath_dir / "QuPath-0.7.0 (console).exe",
            "script",
            "--project",
            project / "project.qpproj",
            "--image",
            self.path.name,
            "--args",
            out_path,
            *processed_args,
            project / "scripts" / f"{script_name}.groovy",
        ]

        # print([str(arg) for arg in args])

        try:
            subprocess.run(
                [str(arg) for arg in args],
                cwd=quPath_dir,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"QuPath Failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise Exception from e

        img = LazyAntsImage(out_path)
        self.maps[script_name] = img
        return img

    def svs_read(self, path: Path) -> ANTsImage:
        level = self.level
        if level == 0:
            print("Loading the image with no downscaling. Your computer may explode!")
            return ants.image_read(str(self.path), *self.args, **self.kwargs)

        slide = tiffslide.TiffSlide(path)

        if level >= len(slide.level_downsamples):
            raise IndexError(f"Selected level not available, select from {list(range(len(slide.level_downsamples)))}")

        target_dims = slide.level_dimensions[level]

        rgba_image = slide.read_region((0, 0), level, target_dims)

        # Convert RGBA to RGB, then to a numpy array
        rgb_image = rgba_image.convert("RGB")
        np_img = np.array(rgb_image)

        # We have (Y, X, C), ANTs expects (X, Y, C)
        np_img = np.transpose(np_img, (1, 0, 2))
        np_img = np.ascontiguousarray(np_img)

        header = self.header_info

        ants_img = ants.from_numpy(
            np_img,
            spacing=header["spacing"][:2],
            origin=header["origin"][:2],
            direction=header["direction"][:2, :2],
            has_components=True,
        )

        return ants_img

    def rotate(self, deg: int) -> ANTsImage:
        if deg == 0:
            return self.img

        img = self.img

        shape = img.shape
        # Pythagoras: determine the diagonal length of the image
        diagonal = math.ceil(math.sqrt(shape[0] ** 2 + shape[1] ** 2))

        # Pad the image equally on all sides to fit the diagonal
        # The diagnoal is the max possible width the image can be when rotating, so if we accomodate for that, the image will never clip
        # We calculate the difference between the diagonal and the current dimensions because ants adds the required size
        pad_x = (diagonal - shape[0]) // 2
        pad_y = (diagonal - shape[1]) // 2

        padded: ANTsImage = ants.pad_image(img, pad_width=[(pad_x, pad_x), (pad_y, pad_y)], value=0)  # type: ignore

        return ants.from_numpy(
            np.array(Image.fromarray(padded.numpy()).rotate(deg)),
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction,
            has_components=img.has_components,
        )

    def __repr__(self):
        return f"LazyAntsImage({self.path})"
