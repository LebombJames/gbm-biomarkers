from __future__ import annotations
import ants
import tiffslide
import os
from pydicom import dcmread
import numpy as np
# from vedo import Volume, show
import vedo
from typing import overload, Literal
from typing_extensions import TypeIs
from pathlib import Path
from __types import *
import gc
from functools import cached_property


class LazyAntsImage():
    """A wrapper around Path that lazy loads an AntsImage when called. Also has some convienience methods to retrieve metadata."""

    def __init__(self, path: Path | None = None, img: ants.ANTsImage | None = None, level: int = 2, *args, **kwargs) -> None:
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
            # Mainly here for type checking, if img was provided in the params but path wasn't, img is returned immediately.
            raise ValueError("No path was provided, and there was no provided image to fallback to.")

        print(f"Loading image: {self.path}")

        return svs_read(self.path, self.level) if self.is_hist else ants.image_read(str(self.path), *self.args, **self.kwargs)

    def is_hist(self):
        return str(self.path).endswith(".svs")

    def header_info(self) -> AntsHeader:
        return ants.image_header_info(str(self.path))

    def metadata(self):
        return ants.read_image_metadata(str(self.path))


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

    volume_data = volume_data * 255.0/volume_data.max()
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
    vol.alpha([
        (0, 0.0),
        (5, 0.0),
        (20, 1.0),
        (25, 1.0)
    ])
    vol.mode(1)
    vedo.show(vol, axes=1, bg="white")


def ants_init(dicom_params: DicomParams, hist_params: HistParams):
    mri_slices: dict[str, LazyAntsImage] = {
        "mri_1": dicom_params["slices"][0],
        "mri_2": dicom_params["slices"][1]
    }

    # # Correctly set depth
    # for slide in mri_slices.values():
    #     slide.set_spacing((slide.spacing[0], slide.spacing[1], 0.5))
    # print(ants.image_header_info(str(dicom_params["slices"][0])))

    # hist_slices = [load_hist(hist_path, hist_params["downsample_svs"]) for hist_path in hist_params["slices"]]
    hist_slices = hist_params["slices"]
    # ants.image_write(
    #     image=hist_slices[0],
    #     filename="out.png"
    # )

    if hist_params["loc_within"]:
        mri_hist_allocation = register_hist_within(hist_slices, hist_params, False)
        if is_registration_dict(mri_hist_allocation):
            mri_hist_allocation = alloc_registration_dict_to_images(mri_hist_allocation)
    else:
        mri_hist_allocation = allocate_hists(list(hist_slices), hist_params["slide_3"])

    stacked_mri_1 = stack_hist(mri_hist_allocation["mri_1"])
    stacked_mri_2 = stack_hist(mri_hist_allocation["mri_2"])

    # coloc_mri_1: RegistrationDict = ants.registration(
    #     fixed=mri_slices["mri_1"],
    #     moving=stacked_mri_1,
    #     verbose=True
    # )

    # coloc_mri_2: RegistrationDict = ants.registration(
    #     fixed=mri_slices["mri_2"],
    #     moving=stacked_mri_2,
    #     verbose=True
    # )

   # print(coloc_mri_1, coloc_mri_2)

   # ants.plot(r)


def svs_read(path: Path, level: int = 1) -> ants.ANTsImage:
    if level == 0:
        return ants.image_read(str(path))

    slide = tiffslide.TiffSlide(path)

    if level >= len(slide.level_downsamples):
        raise Exception(
            f"Selected level not available, select from {list(range(len(slide.level_downsamples)))}"
        )

    target_dims = slide.level_dimensions[level]
    # print(target_dims)

    rgba_image = slide.read_region((0, 0), level, target_dims)

    # Convert RGBA to RGB, then to a numpy array
    rgb_image = rgba_image.convert("RGB")
    np_img = np.array(rgb_image)

    # We have (Y, X, C), ANTs expects (X, Y, C)
    np_img = np.transpose(np_img, (1, 0, 2))
    np_img = np.ascontiguousarray(np_img)

    header: AntsHeader = ants.image_header_info(str(path))
    # print(header)

    ants_img = ants.from_numpy(
        np_img,
        spacing=header["spacing"][:2],
        origin=header["origin"][:2],
        direction=header["direction"][:2, :2],
        has_components=True
    )

    # Multiply the original spacing by the downsample factor
    # effective_spacing_x = spacing[0] * float(target_dims[0])
    # effective_spacing_y = spacing[1] * float(target_dims[1])
    # print(effective_spacing_x, effective_spacing_y)
    # ants_img.set_spacing((effective_spacing_x, effective_spacing_y))

    return ants_img


def allocate_hists(hists: list[LazyAntsImage], slide_3: Literal[0] | Literal[1] | Literal[2] | Literal[3]) -> AllocatedHists[LazyAntsImage]:
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
    hists: HistSlices,
    hist_params: HistParams,
    return_reg_dict: Literal[True]
) -> AllocatedHists[RegistrationDict]: pass


@overload
def register_hist_within(
    hists: HistSlices,
    hist_params: HistParams,
    return_reg_dict: Literal[False]
) -> AllocatedHists[LazyAntsImage]: pass


def register_hist_within(
    hists: HistSlices,
    hist_params: HistParams,
    return_reg_dict: bool = False
) -> AllocatedHists[T]:
    """Register hist slices against others. Returns a dict with hist slides allocated to MRI slides, optionally including the registration info. See `allocate_hists`"""

    registered: AllocatedHists = {
        "mri_1": [],
        "mri_2": []
    }

    fixed = hists[hist_params.get("fixed_image", 2)].img
    moving_dict = allocate_hists(list(hists), hist_params["slide_3"])

    fixed_single_channel = ants.split_channels(fixed)[0]

    print("about to register", fixed)
    for mri_key, imgs in moving_dict.items():
        for img in imgs:

            moving = img.img
            moving_single_channel = ants.split_channels(moving)[0]

            print("img:", moving.numpy().shape)
            reg: RegistrationDict = ants.registration(
                fixed=fixed,
                moving=moving,
                verbose=True,
                type_of_transform="Rigid"
            )
            print("done", reg)
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

    print(reference.spacing, reference.origin)

    spacing = (*reference.spacing, slice_thickness)
    origin = (*reference.origin, z_origin)

    direction = np.eye(3)
    direction[:2, :2] = reference.direction

    print(ants.image_header_info(str(hists[0].path)))

    return ants.from_numpy(
        volume_array,
        spacing=spacing,
        origin=origin,
        direction=direction
    )


def is_registration_dict(dict: AllocatedHists) -> TypeIs[AllocatedHists[RegistrationDict]]:
    """Does this AllocatedHist object contain arrays of RegistrationDicts for each MRI slide?"""
    return hasattr(dict["mri_1"][0], "warpedmovout")


def alloc_registration_dict_to_images(alloc: AllocatedHists[RegistrationDict]) -> AllocatedHists[LazyAntsImage]:
    """Convert an AllocatedHist object from containing arrays of RegistrationDicts for each MRI slide to arrays of images"""
    return {
        "mri_1": [LazyAntsImage(img=reg["warpedmovout"]) for reg in alloc["mri_1"]],
        "mri_2": [LazyAntsImage(img=reg["warpedmovout"]) for reg in alloc["mri_2"]]
    }


# render_dicom_volume()
dicom: DicomParams = {
    "slices": (
        LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm08.dcm", dimension=3),
        LazyAntsImage(Path("GCBA5.23O_SC1__E2") / "MRIm09.dcm", dimension=3)
    )
}
hist: HistParams = {
    "downsample_svs": 1,
    "fixed_image": 2,
    "loc_within": True,
    "slide_3": 3,
    "slices": (
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S1.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S2.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S3.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S4.svs")),
        LazyAntsImage(Path("240920_GCBA_23o_HnE20x_S5.svs"))
    )
}
ants_init(dicom_params=dicom, hist_params=hist)
