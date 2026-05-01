import ants
import tiffslide
import os
import pydicom
import numpy as np
#from vedo import Volume, show
import vedo
import gc
from typing import Tuple

def load_dicom_to_vedo(directory_path: str):
    # 1. Grab ONLY the .dcm files, ignoring everything else
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.dcm')]

    if not files:
        raise ValueError("No .dcm files found in the specified directory.")

    # 2. Read them with pydicom
    slices = [pydicom.dcmread(f) for f in files]

    # 3. Sort them strictly by their Z-axis position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 4. Stack into a 3D Numpy Array
    # pydicom arrays are stacked as (Z, Y, X)
    volume_data = np.stack([s.pixel_array for s in slices])

    volume_data = volume_data * 255.0/volume_data.max()
    print(volume_data[1])

    # 5. Extract the spacing
    dy, dx = slices[0].PixelSpacing # Rows, Columns
    dz = slices[0].SliceThickness

    # 6. Create the Vedo Volume directly from the Numpy array
    vol = vedo.Volume(volume_data)

    # Apply the spacing to ensure the volume isn't distorted
    # We pass it as [dz, dy, dx] to match the numpy array's (Z, Y, X) shape
    vol.spacing([dz, dy, dx])

    return vol

def render_dicom_volume():
    dicom_dir = "GCBA5.23O_SC1__E2" # Ensure this is a string!

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
    vedo.show(vol, axes=1, bg='white')

def ants_init(dicom_path: str, hist_path: str):
    dicom = ants.dicom_read(dicom_path)
    print(type(dicom))

    header = ants.image_header_info(hist_path)
    hist = svs_read(hist_path, header)
    ants.image_write(hist, "out.png")
    print(type(hist))

    print(hist.dimension, dicom.dimension) # blocked: hist is 2d, dicom is 3d, what should i do?

    # This is garbage rn, idk if the data is compatible
    resampled_hist = ants.resample_image_to_target(hist, dicom, verbose=True)
    r = ants.registration(
        fixed=dicom,
        moving=resampled_hist,
        verbose=True
    )
    ants.plot(r)

def svs_read(path: str, header: dict) -> ants.ANTsImage:
    """SVS files are massive, so we should downsample them."""

    slide = tiffslide.TiffSlide(path)

    # Level 0 is the original, each subsequent level is downsampled more
    level = 1

    if level >= len(slide.level_dimensions):
        raise Exception("The selected level does not exist in this .svs file.")

    target_dims = slide.level_dimensions[level]
    print(target_dims)

    rgba_image = slide.read_region((0, 0), level, target_dims)

    # Convert RGBA to RGB, then to a numpy array
    rgb_image = rgba_image.convert('RGB')
    np_img = np.array(rgb_image)

    # We have (Y, X, C), ANTs expects (X, Y, C)
    np_img = np.transpose(np_img, (1, 0, 2))
    np_img = np.ascontiguousarray(np_img)

    print(header)
    ants_img = ants.from_numpy(
        np_img,
        spacing = header["spacing"][:2],
        origin = header["origin"][:2],
        direction = header["direction"][:2, :2],
        has_components=True
    )

    # Multiply the original spacing by the downsample factor
    # effective_spacing_x = spacing[0] * float(target_dims[0])
    # effective_spacing_y = spacing[1] * float(target_dims[1])
    # print(effective_spacing_x, effective_spacing_y)
    # ants_img.set_spacing((effective_spacing_x, effective_spacing_y))

    return ants_img


# render_dicom_volume()
ants_init(dicom_path="GCBA5.23O_SC1__E2", hist_path="240920_GCBA_23o_HnE20x_S1.svs")
