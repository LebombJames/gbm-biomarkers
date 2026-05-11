from __future__ import annotations
import ants
import tiffslide
import os
from pydicom import dcmread
import numpy as np

# from vedo import Volume, show
import vedo
from typing import overload, Literal, cast, no_type_check
from typing_extensions import TypeIs
from pathlib import Path
from __types import *
import gc
from functools import cached_property

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class LazyAntsImage:
    """
    A wrapper around Path that lazy loads an AntsImage when called, or returns a cached version.

    Also has some convenience properties to retrieve metadata.
    """

    def __init__(
        self,
        path: Path | None = None,
        img: ants.ANTsImage | None = None,
        level: int = 1,
        *args,
        **kwargs,
    ) -> None:
        self.path = path
        self.level = level
        self.args = args
        self.kwargs = kwargs

        if img is not None:
            self.img = img

    @cached_property
    def img(self) -> ants.ANTsImage:
        """Return the cached Ants image, or load it from the path if not loaded yet."""

        if self.path is None:
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately before
            # this whole function is even called.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        print(f"Loading image: {self.path}")

        return (
            self.svs_read(self.path, self.level)
            if self.is_hist
            else ants.image_read(str(self.path), *self.args, **self.kwargs)
        )

    @property
    def is_hist(self):
        return str(self.path).endswith(".svs")

    @property
    def header_info(self) -> AntsHeader:
        if self.path is None and self.img is not None:
            raise ValueError("Header Info is not available for in memory images.")

        return ants.image_header_info(str(self.path))

    @property
    def metadata(self):
        if self.path is None and self.img is not None:
            raise ValueError("Metadata Info is not available for in memory images.")

        return ants.read_image_metadata(str(self.path))

    @cached_property
    def greyscale_img(self) -> ants.ANTsImage:
        """Returns a greyscale image constructed from the means of the RGB channels, and inverts it (255 becomes 0)"""
        img = self.img

        if img.components <= 1:
            return img

        img_np: npt.NDArray[np.float64] = img.numpy()

        # The mean of RGB values is grayscale
        gray_np = np.mean(img_np, axis=-1, dtype=np.float64)

        # Invert. Ants seems to work best with a black background
        inverted = gray_np.max() - gray_np

        return ants.from_numpy(inverted, origin=img.origin, spacing=img.spacing, direction=img.direction)

    def svs_read(self, path: Path, level: int = 1) -> ants.ANTsImage:
        if level == 0:
            return ants.image_read(str(path))

        slide = tiffslide.TiffSlide(path)

        if level >= len(slide.level_downsamples):
            raise IndexError(f"Selected level not available, select from {list(range(len(slide.level_downsamples)))}")

        target_dims = slide.level_dimensions[level]
        # print(target_dims)

        rgba_image = slide.read_region((0, 0), level, target_dims)

        # Convert RGBA to RGB, then to a numpy array
        rgb_image = rgba_image.convert("RGB")
        np_img = np.array(rgb_image)

        # We have (Y, X, C), ANTs expects (X, Y, C)
        np_img = np.transpose(np_img, (1, 0, 2))
        np_img = np.ascontiguousarray(np_img)

        header = self.header_info
        # print(header)

        ants_img = ants.from_numpy(
            np_img,
            spacing=header["spacing"][:2],
            origin=header["origin"][:2],
            direction=header["direction"][:2, :2],
            has_components=True,
        )

        return ants_img

    def __repr__(self):
        return ants.ANTsImage.__repr__(self.img)


def load_dicom_to_vedo(directory_path: str):
    # 1. Grab ONLY the .dcm files, ignoring everything else
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".dcm")]

    if not files:
        raise ValueError("No .dcm files found in the specified directory.")

    # 2. Read them with pydicom
    slices = [dcmread(f) for f in files]

    # 3. Sort them strictly by their Z-axis position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 4. Stack into a 3D Numpy Array
    # pydicom arrays are stacked as (Z, Y, X)
    volume_data = np.stack([s.pixel_array for s in slices])

    volume_data = volume_data * 255.0 / volume_data.max()
    print(volume_data[1])

    # 5. Extract the spacing
    dy, dx = slices[0].PixelSpacing  # Rows, Columns
    dz = slices[0].SliceThickness

    # 6. Create the Vedo Volume directly from the Numpy array
    vol = vedo.Volume(volume_data)

    # Apply the spacing to ensure the volume isn't distorted
    # We pass it as [dz, dy, dx] to match the numpy array's (Z, Y, X) shape
    vol.spacing([dz, dy, dx])

    return vol


def render_dicom_volume():
    dicom_dir = "GCBA5.23O_SC1__E2"  # Ensure this is a string!

    # Load the volume
    vol = load_dicom_to_vedo(dicom_dir)
    print("Volume successfully built from Numpy array!")

    # Style and render
    vmin, vmax = 1, 68
    vol.cmap("bone")
    vol.alpha([(0, 0.0), (5, 0.0), (20, 1.0), (25, 1.0)])
    vol.mode(1)
    vedo.show(vol, axes=1, bg="white")


def explore_3D_array(arr: npt.NDArray):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.2)

    # Set the starting slice (e.g., the middle of the volume)
    initial_slice = arr.shape[2] // 2

    im = ax.imshow(arr[:, :, initial_slice], cmap="gray")
    ax.set_title(f"Slice: {initial_slice}")
    ax.axis("off")

    # 3. Create a dedicated set of axes just for the slider
    # Dimensions are [left, bottom, width, height] in figure percentages
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

    # 4. Initialize the native Matplotlib Slider
    slice_slider = Slider(
        ax=ax_slider, label="Z-Slice", valmin=0, valmax=arr.shape[2] - 1, valinit=initial_slice, valstep=1
    )

    def update(val):
        z_index = int(slice_slider.val)

        # Update the existing image data instead of creating a new plot
        im.set_data(arr[:, :, z_index])
        ax.set_title(f"Slice: {z_index}")

        # Tell matplotlib to redraw the figure
        fig.canvas.draw_idle()

    # 6. Connect the slider movement to the update function
    slice_slider.on_changed(update)

    # 7. CRITICAL FIX: Attach the slider to the figure so it isn't garbage collected
    # fig.slider_obj = slice_slider

    plt.show()


def ants_init(dicom_params: DicomParams, hist_params: HistParams):
    mri_slices: dict[str, LazyAntsImage] = {
        "mri_1": dicom_params["slices"][0],
        "mri_2": dicom_params["slices"][1],
    }

    # hist_slices = [load_hist(hist_path, hist_params["downsample_svs"]) for hist_path in hist_params["slices"]]
    hist_slices = list(hist_params["slices"])

    # image = hist_slices[1]
    # reference = hist_slices[2]

    # rescale: ants.ANTsImage = ants.resample_image_to_target(
    #     image=reference.greyscale_img, target=image.greyscale_img, interp_type="nearestNeighbor"
    # )  # type: ignore

    # print("before:", reference)
    # print("target", image)
    # print("rescale:", rescale)

    # ants.image_write(image=rescale.astype("uint8"), filename="rescaled.png")

    # # print(hist_slices[1].header_info, hist_slices[2].header_info)
    # cropped = LazyAntsImage(img=ants.crop_image(image.greyscale_img, rescale))

    # ants.image_write(image=cropped.img, filename="cropped.png")

    # ants.image_write(image=hist_slices[0].img, filename="out1.png")
    # ants.image_write(image=hist_slices[3].img, filename="out4.png")
    # ants.image_write(image=hist_slices[4].img, filename="out5.png")

    if hist_params["loc_within"]:
        mri_hist_allocation = register_hist_within(hist_slices, hist_params, False)  # type: ignore
    else:
        mri_hist_allocation = allocate_hists(list(hist_slices), hist_params["slide_3"])

    hists = [*mri_hist_allocation["mri_1"], *mri_hist_allocation["mri_2"]]

    stacked_mri_1 = stack_hist(mri_hist_allocation["mri_1"])
    stacked_mri_2 = stack_hist(mri_hist_allocation["mri_2"])

    coloc_mri_1: RegistrationDict = ants.registration(fixed=mri_slices["mri_1"].img, moving=stacked_mri_1)

    coloc_mri_2: RegistrationDict = ants.registration(fixed=mri_slices["mri_2"].img, moving=stacked_mri_2)

    explore_3D_array(coloc_mri_1["warpedmovout"].numpy())
    explore_3D_array(coloc_mri_2["warpedmovout"].numpy())


# print(coloc_mri_1, coloc_mri_2)

# ants.plot(r)


def allocate_hists(hists: list[LazyAntsImage], slide_3: Literal[0, 1, 2, 3]) -> AllocatedHists[LazyAntsImage]:
    """Assign each hist slide an MRI slide to be coregistered with (assuming 5 hist slides and 2 MRI slides)"""
    dict = {}
    match slide_3:
        case 0:
            dict: AllocatedHists = {"mri_1": hists[0:1], "mri_2": hists[3:4]}
        case 1:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[3:4]}
        case 2:
            dict: AllocatedHists = {"mri_1": hists[0:1], "mri_2": hists[2:4]}
        case 3:
            dict: AllocatedHists = {"mri_1": hists[0:2], "mri_2": hists[2:4]}
    return dict


@overload
def register_hist_within(
    hists: HistSlices, hist_params: HistParams, return_reg_dict: Literal[True]
) -> AllocatedHists[RegistrationDict]:
    pass


@overload
def register_hist_within(
    hists: HistSlices, hist_params: HistParams, return_reg_dict: Literal[False]
) -> AllocatedHists[LazyAntsImage]:
    pass


def register_hist_within(
    hists: HistSlices, hist_params: HistParams, return_reg_dict: bool = False
) -> AllocatedHists[T]:
    """Register hist slices against others. Returns a dict with hist slides allocated to MRI slides, optionally including the registration info. See `allocate_hists`"""

    registered: AllocatedHists = {"mri_1": [], "mri_2": []}

    fixed = hists[hist_params.get("fixed_image", 2)].greyscale_img
    moving_dict = allocate_hists(list(hists), hist_params["slide_3"])

    for mri_key, imgs in moving_dict.items():
        for i, img in enumerate(imgs):

            moving = img.greyscale_img

            # ants.image_write(image=fixed.astype("uint8"), filename="fixed.png")
            # ants.image_write(image=moving.astype("uint8"), filename="moving.png")

            reg: RegistrationDict = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")

            # ants.image_write(image=reg["warpedmovout"].astype("uint8"), filename=f"{mri_key}{i}.png")
            registered[mri_key].append(reg if return_reg_dict else LazyAntsImage(img=reg["warpedmovout"]))
            gc.collect()
    return registered


def stack_hist(hists: list[LazyAntsImage]) -> ants.ANTsImage:
    """Create a 3d volume from hist slices"""
    arrays = [img.img.numpy() for img in hists]

    volume_array = np.stack(arrays, axis=-1)

    # Rebuild metadata to include a 3rd element in the tuple for the Z axis.
    reference = hists[0].img

    slice_thickness = 0.08  # 80 microns
    z_origin = 0.0

    spacing = (*reference.spacing, slice_thickness)
    origin = (*reference.origin, z_origin)

    direction = np.eye(3)
    direction[:2, :2] = reference.direction

    return ants.from_numpy(volume_array, spacing=spacing, origin=origin, direction=direction)


def is_registration_dict(
    dict: AllocatedHists,
) -> TypeIs[AllocatedHists[RegistrationDict]]:
    """Does this AllocatedHist object contain arrays of RegistrationDicts for each MRI slide?"""
    return hasattr(dict["mri_1"][0], "warpedmovout")


def alloc_registration_dict_to_images(
    alloc: AllocatedHists[RegistrationDict],
) -> AllocatedHists[LazyAntsImage]:
    """Convert an AllocatedHist object from containing arrays of RegistrationDicts for each MRI slide to arrays of images"""
    return {
        "mri_1": [LazyAntsImage(img=reg["warpedmovout"]) for reg in alloc["mri_1"]],
        "mri_2": [LazyAntsImage(img=reg["warpedmovout"]) for reg in alloc["mri_2"]],
    }


# render_dicom_volume()
dicom: DicomParams = {
    "slices": (
        LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm08.dcm", dimension=3),
        LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm09.dcm", dimension=3),
    )
}
hist: HistParams = {
    "fixed_image": 2,
    "loc_within": True,
    "slide_3": 3,
    "slices": (
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S1.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S2.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S3.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S4.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S5.svs")),
    ),
}
ants_init(dicom_params=dicom, hist_params=hist)
