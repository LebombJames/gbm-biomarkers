
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


def explore_3D_array(arr: np.ndarray[tuple[int,int,int], np.dtypes.Float64DType]):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.2)

    # Set the starting slice (e.g., the middle of the volume)
    initial_slice = 0

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

def brain_extraction_nirodents(mri: LazyAntsImage):

    client = docker.from_env()

    templateflow_host = os.path.expanduser("~/.cache/templateflow")
    os.makedirs(templateflow_host, exist_ok=True)

    volumes = {
        os.getcwd(): {"bind": "/data", "mode": "ro"},
        f"{os.getcwd()}/out": {"bind": "/out", "mode": "rw"},
        templateflow_host: {"bind": "/templateflow", "mode": "rw"},
    }

    # Not strictly necessary since run also pulls, but it creates console output when it's done
    image = client.images.pull("nipreps/nirodents:0.2.1")

    try:
        container = cast(
            docker.models.containers.Container,
            client.containers.run(
                image=image,
                command=f"/data/{mri.path.as_posix() if isinstance(mri.path, Path) else ''} -o /out/ -w /out/work",
                volumes=volumes,
                environment={"TEMPLATEFLOW_HOME": "/templateflow"},
                remove=True,
                detach=True,
            ),
        )

        for log_line in container.logs(stream=True):
            print(log_line.decode("utf-8"), end="")

        result = container.wait()
        print(f"\nContainer exited with code: {result['StatusCode']}")

    except docker.errors.ContainerError as e:
        print(e.stderr.decode("utf-8") if e.stderr else e.container.logs().decode("utf-8"))
    except Exception as e:
        raise Exception from e

def stack_hist(hists: list[LazyAntsImage]) -> ANTsImage:
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
        "mri_1": [LazyAntsImage(reg["warpedmovout"]) for reg in alloc["mri_1"]],
        "mri_2": [LazyAntsImage(reg["warpedmovout"]) for reg in alloc["mri_2"]],
    }

def center_and_pad(threshold_dict: ThresholdDict, target_shape: tuple[int, int]) -> ThresholdDict:
    """
    Center the subject and set constant padding around the perimeter of its boundingbox

    Args:
        threshold_dict (ThresholdDict): The threshold data with which to determine the bounding box
        pad_width (int or tuple of tuples of ints): The amount to pad by. See `np.pad`.

    Returns:
        ANTsImage: The centered and padded image
    """
    img = threshold_dict["img"]
    region = threshold_dict["region"]
    min_x, min_y, max_x, max_y = region.bbox

    x_slice = slice(min_x, max_x)
    y_slice = slice(min_y, max_y)

    bbox_crop: ANTsImage = img[x_slice, y_slice]  # type: ignore
    bbox_crop.astype("uint8").to_file("centered.png")

    arr = bbox_crop.numpy()
    w, h = arr.shape[:2]

    target_aspect_ratio = target_shape[0] / target_shape[1]
    current_aspect_ratio = w / h

    # Calculate how much total padding is needed
    if current_aspect_ratio < target_aspect_ratio:
        pad_w = int(h * target_aspect_ratio) - w
        pad_h = 0
    else:
        pad_h = int(w / target_aspect_ratio) - h
        pad_w = 0

    # Split padding equally between left/right and top/bottom
    pad_w_left, pad_w_right = pad_w // 2, pad_w - (pad_w // 2)
    pad_h_top, pad_h_bot = pad_h // 2, pad_h - (pad_h // 2)

    new_start_x = int(min_x - pad_w_left)
    new_start_y = int(min_y - pad_h_top)

    new_origin = ants.transform_index_to_physical_point(img, (new_start_x, new_start_y))
    print(new_origin)

    padded_arr = np.pad(
        arr, pad_width=((pad_w_left, pad_w_right), (pad_h_top, pad_h_bot)), mode="constant", constant_values=0
    )

    padded = ants.from_numpy(
        padded_arr, origin=tuple(new_origin), spacing=img.spacing, direction=img.direction
    )  # type: ignore

    mask_crop = threshold_dict["mask"][x_slice, y_slice].astype(np.float32).copy()
    padded_arr = np.pad(
        mask_crop, pad_width=((pad_w_left, pad_w_right), (pad_h_top, pad_h_bot)), mode="constant", constant_values=0
    )
    mask_ants = ants.from_numpy(
        padded_arr, spacing=padded.spacing, origin=padded.origin, direction=padded.direction, has_components=False
    )

    largest = measure.regionprops(padded_arr.astype(int))[0]

    print(padded.shape, mask_ants.shape)

    return {
        "img": padded,
        "mask": mask_ants,
        "region": largest,
    }

async def run_qupath_script_async(self, script_name: str, out_filename: str = "") -> ANTsImage:
        """
        Run a qupath script that takes this image as an argument (`args[0]` in the groovy script), and creates an output image, which we read.
        """
        if self.path is None:
            raise ValueError("Scripts not available for in memory images.")

        print(os.getcwd())
        home = Path.home()
        quPath_dir = home / "AppData" / "Local" / "QuPath-0.7.0"

        cwd = Path(os.getcwd())
        out_path = cwd / f"{out_filename or self.path}-{script_name}.tif"
        project = cwd / "HnE" / "project"

        qupath_exe = (quPath_dir / "QuPath-0.7.0 (console).exe",)

        args = [
            qupath_exe,
            "script",
            "--project",
            project / "project.qpproj",
            "--image",
            self.path.name,
            "--args",
            out_path,
            project / "scripts" / f"{script_name}.groovy",
        ]

        str_args = [str(arg) for arg in args]

        # 1. Start the subprocess
        process = await asyncio.create_subprocess_exec(
            str(qupath_exe),
            *str_args[1:],
            cwd=quPath_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 2. Define a helper to stream lines as they arrive
        async def stream_output(stream: asyncio.StreamReader, prefix: str):
            while True:
                line = await stream.readline()
                if not line:
                    break  # EOF reached

                # Print live to the console.
                # end="" is used because `line` already contains the newline character.
                print(f"{prefix}{line.decode()}", end="")

        # 3. Read stdout, read stderr, and wait for the process to exit concurrently
        if process.stderr is None or process.stdout is None:
            raise AttributeError("process has no stdout or stderr somehow!")

        await asyncio.gather(
            stream_output(process.stdout, "[QuPath INFO] "),
            stream_output(process.stderr, "[QuPath ERR]  "),
            process.wait(),
        )

        # 4. Check exit status
        if process.returncode != 0:
            raise RuntimeError(f"QuPath script failed with return code {process.returncode}.")

        # 5. Read the image asynchronously
        loop = asyncio.get_running_loop()
        img: ANTsImage = await loop.run_in_executor(None, ants.image_read, str(out_path))  # type: ignore

        self.maps[script_name] = img
        return img

def necrosis_correct_cellularity(cellularity: ANTsImage, necrosis: ANTsImage) -> ANTsImage:
    """
    Areas with high necrosis can cause low cellularity, despite still having high tumour infiltration.
    We can correct by dividing the cellularity map by the inverted necrosis map.
    Any areas with no necrosis remain unchanged (`x/1=x`).
    Areas with necrosis are made expontentially more intense (`x/0.5=2x`, `x/0.1=10x`)

    Args:
        cellularity: The cellularity map
        necrosis: The necrosis map

    Returns:
        ANTsImage: The necrosis-corrected cellularity map
    """
    ants.copy_image_info(cellularity, necrosis)

    necrosis_inverted = necrosis.max() - necrosis

    return cellularity / necrosis_inverted

def best_rotation(hist: LazyAntsImage, mri: ANTsImage):
    out = {"img": None, "mi": float("-inf"), "rotation": 0}

    histimg = hist.greyscale_img()
    mri_sliced: ANTsImage = ants.slice_image(mri, axis=-1, idx=0).astype("float32")  # type: ignore

    mri_processed = prepare_mri(mri_sliced)["img"]

    hist_processed: ANTsImage = scale_and_align_to_ref(histimg, mri_processed, resample=True)  # type: ignore

    for deg in range(0, 360):
        print(deg)
        rotated = LazyAntsImage(hist_processed).rotate(deg)
        mi = ants.image_mutual_information(rotated, mri_processed) * -1  # MI is negative for some reason

        if mi > out["mi"]:
            out["img"] = rotated
            out["mi"] = mi
            out["rotation"] = deg

    out["mri"] = mri_processed
    return out