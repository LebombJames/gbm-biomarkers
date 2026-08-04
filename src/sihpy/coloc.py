from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import gc
from copy import deepcopy
from functools import partial
from pathlib import Path

import ants
import numpy as np
from ants import ANTsImage
from pandas import DataFrame

from scripts.stats import combine_maps_integral
from src.sihpy.__types import *
from src.sihpy.config import DEBUG
from src.sihpy.hist import (
    allocate_hists,
    create_hist_volume,
    prepare_hist,
    register_hist_within,
    transform_original_hist,
)
from src.sihpy.img_utils import create_checkerboard, prepare_mri
from src.sihpy.LazyAntsImage import LazyAntsImage
from src.sihpy.maps import combine_maps, process_maps
from src.sihpy.plots import collate_checkerboard_plots, plot_roi_intensity
from src.sihpy.utils import ensure_path_exists, func_timer, pretty_hist_filename, pretty_mri_key


def run_registration_in_thread(
    dicom_params: DicomParams, hist_params: HistParams, reg_params: dict[str, Any], run_name="coloc", strict=True
):
    """
    Helper function to be create a task object to be used with `run_many_registrations`.
    """
    return {
        "dicom_params": dicom_params,
        "hist_params": hist_params,
        "reg_params": reg_params,
        "run_name": run_name,
        "strict": strict,
    }


async def run_many_registrations(jobs: list[dict[str, Any]], max_workers: int = 3) -> list[RegPlots | None]:
    """
    Run a series of registrations across multiple threads, improving performance by avoiding the GIL.

    Args:
        jobs (list of dicts): Task dicts produced by `run_registration_in_thread`
        max_workers (int): No more than this number of workers may run at once. Lower this number if you experience out of memory errors.

    Example:
    ```python
    tasks = [
        run_registration_in_thread(mri1, hist1, reg1, "coloc1"),
        run_registration_in_thread(mri2, hist2, reg2, "coloc2"),
        ...
    ]

    plots = run_many_registrations(tasks, max_tasks=3)
    ```
    """
    loop = asyncio.get_running_loop()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        tasks = []

        for job_kwargs in jobs:
            func = partial(run_registration, **job_kwargs)

            task = loop.run_in_executor(pool, func)
            tasks.append(task)

        return await asyncio.gather(*tasks)


def run_registration(
    dicom_params: DicomParams, hist_params: HistParams, reg_params: dict[str, Any], run_name="coloc", strict=True
) -> RegPlots | None:
    """
    Run a registration between the MRIs in `dicom_params` and the histologies in `hist_params`.
    This is the main function end-users should interact with (or the multi-threaded equivalents)

    Args:
        dicom_params (DicomParams): MRI images and parameters
        hist_params (HistParams): Histology (and map) images and parameters
        reg_params (dict[str, Any]): Parameters for `ants.registration`. Also includes `out_prefix`, which determines the output files.
        run_name (str, optional): Name of this registration, used in diagnostic plots and to distinguish runs across threads. Defaults to "coloc".
        strict (bool, optional): Throw if a registration fails, normally due to a very poor overlap. Defaults to True.

    Raises:
        RuntimeError: _description_

    Returns:
        RegPlots | None: _description_
    """
    from src.sihpy.utils import progress

    mri_slices = dicom_params["slices"]
    hist_slices = hist_params["slices"]

    out_maps: list[ProcessedMap] = []
    plots: RegPlots = {
        "name": run_name,
        "checkerboard": [],
        "map_overview": [],
        "transformed_original": [],
        "mri_overview": [],
    }
    map_df_list: list[dict[str, str | float]] = []
    failures = 0

    base_out_path = Path("out") / reg_params["out_prefix"]

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(hist_slices, hist_params, dicom_params, return_reg_dict=False)
    else:
        # Group the histology based on which MRI slice they align to
        hist_allocation = allocate_hists(hist_slices, dicom_params)

    # progress.reset(total=sum(len(v) for v in hist_allocation.values()))
    # print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

    for mri_key, mri_dict in mri_slices.items():
        map_img = mri_dict["img"]

        # Make any (X,Y,1) images (X,Y)
        mri_zero: ANTsImage = ants.slice_image(map_img.img, axis=-1, idx=0)  # type: ignore
        mri_zero.set_direction(np.eye(2))  # Sanity check

        # Process MRI
        progress.write(f"{run_name} | Processing {mri_key}")
        mri_prepared_dict = prepare_mri(mri_zero, base_out_path / "MRI" / f"{mri_key}_overview.png")
        mri_processed = mri_prepared_dict["img"]
        mri_mask = mri_prepared_dict["mask"]

        plots["mri_overview"].append(
            {
                "img": {"before": mri_zero, "after": mri_processed},
                "mri_key": mri_key,
                "hist_name": None,
                "animal_name": str(map_img.flags.get("path", "").name),
            }
        )

        if DEBUG:
            mri_processed.to_file(ensure_path_exists(base_out_path / "MRI" / f"{mri_key}-processed.nii.gz"))

        for slice_details in hist_allocation.get(mri_key, []):
            progress.write(f"{run_name} | Processing histology {slice_details['img'].path.name}")

            # Get greyscale Histology image and process
            hist_zero = slice_details["img"].greyscale_img(hist_params["greyscale_type"])
            hist_out_path = base_out_path / slice_details["img"].path.name / mri_key

            hist_processed = prepare_hist(
                hist_zero, slice_details, mri_processed, mri_mask, center=False, resample=True
            )
            hist_img = hist_processed["img"]
            hist_mask = hist_processed["mask"]

            if DEBUG:
                hist_img.to_file(ensure_path_exists(hist_out_path / f"final_hist.ome.tif"))
                (hist_mask * 255).to_file(ensure_path_exists(hist_out_path / f"final_mask.ome.tif"))

            affine_init = None
            if reg_params["use_initial_affine"]:
                progress.write(f"{run_name} | Generating affine initializer")
                affine_init = [
                    ants.affine_initializer(  # type: ignore
                        fixed_image=mri_processed,
                        moving_image=hist_img,
                        mask=mri_mask,
                        radian_fraction=1.0,
                        search_factor=360,
                    )
                ]

            progress.write(f"{run_name} | Running registration")
            try:
                registered: RegistrationDict = ants.registration(  # type: ignore
                    fixed=mri_processed,
                    moving=hist_img,
                    mask=mri_mask,
                    moving_mask=hist_mask,
                    initial_transform=affine_init,
                    **reg_params,
                )
            except RuntimeError as e:
                if strict:
                    raise RuntimeError(
                        f"{run_name} | Registration failed. {mri_key}, hist: {slice_details['img'].path.name}"
                        f"\n MRI dims: {mri_processed.shape}, Hist dims: {hist_img.shape}"
                        f"\n MRI unique {np.unique(mri_processed.view())}, hist unique: {np.unique(hist_img.view())}"
                    ) from e
                else:
                    progress.write(
                        f"{run_name} | Registration failed. {mri_key}, hist: {slice_details['img'].path.name}"
                        f"\n MRI dims: {mri_processed.shape}, Hist dims: {hist_img.shape}"
                        f"\n MRI unique {np.unique(mri_processed.view())}, hist unique: {np.unique(hist_img.view())}"
                        "\n Strict is set to false. Continuing!"
                    )
                    failures += 1
                    continue

            if DEBUG:
                registered["warpedmovout"].to_file(ensure_path_exists(hist_out_path / f"registered_hist.ome.tif"))

            progress.write(f"{run_name} | Transforming the original high-res histology")
            transformed_original_hist = transform_original_hist(
                hist_zero, slice_details, mri_processed, mri_mask, registered["fwdtransforms"], out_path=hist_out_path
            )
            plots["transformed_original"].append(
                {"mri_key": mri_key, "hist_name": slice_details["img"].path.name, "img": transformed_original_hist}
            )

            transformed_hist_mask: ANTsImage = ants.apply_transforms(
                transformlist=registered["fwdtransforms"],
                fixed=hist_mask,
                moving=hist_mask,
                interpolator="genericLabel",
            )  # type: ignore

            labels = ants.label_overlap_measures(transformed_hist_mask, mri_mask)  # type: ignore
            labels.rename(
                columns={"UnionOverlap": "UnionOverlap (Jaccard)", "MeanOverlap": "MeanOverlap (Dice)"}, inplace=True
            )
            labels.to_csv(hist_out_path / "overlap_measures.csv", index=False)
            # UnionOverlap = Jaccard, MeanOverlap = Dice. TotalOrTargetOVerlap = Target overlap

            if slice_details["maps"]:

                if DEBUG:
                    (transformed_hist_mask * 255).to_file(
                        ensure_path_exists(hist_out_path / f"transformed_mask.ome.tif")
                    )

                maps = process_maps(
                    slice_details,
                    registered["fwdtransforms"],
                    mri_processed,
                    hist_mask,
                    mri_key=mri_key,
                    dicom_params=dicom_params,
                    hist_params=hist_params,
                    out_path=hist_out_path,
                )

                for map_name, map_dict in maps.items():
                    progress.write(
                        f"{run_name} | Map {map_name} computed with MI: {map_dict['mutual_info']}. (Control: {map_dict['control_mi']})"
                    )
                    out_maps.append(map_dict)

                    # if pretty_hist_filename(slice_details["img"].path.name) == "23R-1":
                    plots["map_overview"].append(
                        {
                            "img": plot_roi_intensity(
                                map_registered=map_dict["img"].img,
                                mri=mri_processed,
                                roi=((40, 100), (115, 140)),
                                mi=map_dict["mutual_info"],
                                title=pretty_hist_filename(slice_details["img"].path.name),
                                out_path=hist_out_path / "maps" / map_name / "overview.png",
                                mri_title=pretty_mri_key(mri_key),
                            ),
                            "mri_key": mri_key,
                            "hist_name": slice_details["img"].path.name,
                            "map_name": map_name,
                        }
                    )

                    map_df_list.append(
                        {
                            "hist_name": pretty_hist_filename(slice_details["img"].path.name),
                            "map_name": map_name,
                            "mi": map_dict["mutual_info"],
                            "mri_key": mri_key,
                            "no_reg_mi": map_dict["control_mi"],
                        }
                    )

            plots["checkerboard"].append(
                {
                    "img": create_checkerboard(
                        mri_processed,
                        transformed_original_hist,
                        squares=(16, 16),
                        mri_mask=mri_mask,
                        hist_mask=transformed_hist_mask,
                        out_path=hist_out_path,
                    ),
                    "mri_key": mri_key,
                    "hist_name": slice_details["img"].path.name,
                }
            )

            create_hist_volume({mri_key: registered["warpedmovout"]}, dicom_params, out_path=hist_out_path)

            del slice_details["img"].img
            progress.update(1)
            progress.write("---")
            gc.collect()

    progress.write("Finished registrations. Creating diagnostic outputs.")

    combined = combine_maps(out_maps)

    combine_maps_integral(out_maps, dicom_params["slices"])

    map_df = DataFrame(map_df_list, copy=False)
    try:
        map_df.to_csv(ensure_path_exists(Path(base_out_path) / "maps.csv"), index=False)
    except PermissionError:
        progress.write(f"{Path(base_out_path) / 'maps.csv'} cannot be written, is it currently open?")

    for map_key, map_group in combined.items():
        create_hist_volume(
            map_group,
            dicom_params,
            out_path=base_out_path / "sih_maps" / map_key / "combined.nii.gz",
            interp="nearestNeighbor",
            is_map=True,
        )
        for mri_key, map_img in map_group.items():
            create_hist_volume(
                {mri_key: map_img},
                dicom_params,
                out_path=base_out_path / "sih_maps" / map_key / f"{mri_key}-{map_key}.nii.gz",
                interp="nearestNeighbor",
                is_map=True,
            )
            # mri_img.to_file(str(base_out_path / map_key / f"{mri_key}-{map_key}.nii.gz"))

    if failures:
        progress.write(f"{run_name} | Registration complete with {failures} failures!")

    gc.collect()
    # collate_checkerboard_plots(plots["checkerboard"], run_name, out_path=base_out_path)
    # collate_mri_plots(plots["mri_overview"], run_name, out_path=base_out_path)
    # collate_transformed_originals(plots["transformed_original"], run_name, out_path=base_out_path)
    # collate_map_plots(plots["map_overview"], run_name, out_path=base_out_path)
    # return plots


def generate_maps_for_params(hist_params: HistParams, scripts: dict[str, ScriptDict]) -> HistParams:
    """
    Run qupath scripts for each slice in the hist params

    Args:
        hist_params (HistParams): The hist params to run scripts for
        scripts (dict[str, str]): Key: the name of the resulting map (ie the key in the "maps" dict in the return value).
        Value: the name of the qupath script (see LazyAntsImage.run_qupath_script)

    Returns:
        HistParams: The updated hist params with the added maps
    """

    for slice_entry in hist_params["slices"]:

        for map_name, script_dict in scripts.items():
            map_output = slice_entry["img"].run_qupath_script(
                project_path=Path("23yqp"), script_name=script_dict["script_name"], script_args=script_dict["args"]
            )

            if map_name in slice_entry["maps"]:
                slice_entry["maps"][map_name]["map_img"] = map_output
            else:
                slice_entry["maps"][map_name] = {
                    "map_img": map_output,
                    "necrosis_correct": False,  # Sensible default
                    "combine_type": "add",  # Sensible default
                }

    return hist_params


def build_dicom_params(path: Path, slices_idx: list[int]) -> DicomParams:
    """
    Given a path to a folder of dicoms, create a DicomParams object

    Args:
        path (Path): The path to a folder of dicoms
        slices_idx (list of ints): List indices of slices (starting at 0) that correspond to the histology
          to be used in the registration. E.g 7 = MRIm08.dcm
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist!")

    out: DicomParams = {"slices": {}, "volume": ants.dicom_read(str(path))}

    volume: ANTsImage = out["volume"]

    for i, slice_idx in enumerate(slices_idx, start=1):
        sliced: ANTsImage = ants.slice_image(volume, axis=-1, idx=slice_idx)  # type: ignore
        sliced_np = sliced.numpy()[..., np.newaxis]

        sliced_ants = ants.from_numpy(
            sliced_np, spacing=volume.spacing, origin=volume.origin, direction=volume.direction
        )

        # LazyAntsImage doesn't support both an img and a path, so record the path in flags
        out["slices"][f"mri_{i}"] = {"img": LazyAntsImage(sliced_ants, flags={"path": path}), "index": slice_idx}

    return out


def build_hist_slices(
    path: Path,
    map_dir: Path | None = None,
    necrosis_dir_name: Path | None = None,
    params: dict[int, dict[str, Any]] | None = None,
) -> list[HistSlicesDict]:
    """
    Create a list of HistSlicesDicts based on histology files and maps in a folder.

    Expected folder structure:
    ```
    root/
    ├── img1.svs
    ├── img2.svs
    ├── ...
    └── maps/
        ├── cell_count/
        │   ├── img1.tif
        │   ├── img2.tif
        │   └── ...
        ├── cell_density/
        │   ├── img1.tif
        │   ├── img2.tif
        │   └── ...
        ├── ...
        └── necrosis/
    ```
    Args:
        path (Path): The path to `root/`, as in the diagram above.

        map_dir (Path, optional): The path to `maps/` as in the diagram above, defaults to `root/maps`.

        necrosis_dir_name (Path, optional): The name of the necrosis directory within `maps`, relative to `maps`. Defaults to `necrosis`.

        params (dict[int, Any], optional): Elements of HistSlicesDict to apply to the output dict of the element at the array index corresponding to key. Files read through `path` are sorted alphabetically, so the order of the return list is always the same. E.g `{3: {"rotation": 100}}` will set the rotation of the element at array index 3 (starts at 0!).
    """
    out: list[HistSlicesDict] = []

    if not params:
        params = {}

    imgs = list(path.glob("*.svs"))

    if len(imgs) == 0:
        raise FileNotFoundError(f"No .svs. files in {path}")

    maps = map_dir or path / "maps"
    if not maps.exists():
        print(f"No maps folder found in {path}, continuing without processing maps.")
        maps = None

    # Sort to ensure identical order between runs/machines
    for i, img in enumerate(sorted(imgs)):
        idx_params = params.get(i, {})

        obj: HistSlicesDict = {
            "img": idx_params.get("img", LazyAntsImage(img)),
            "rotation": idx_params.get("rotation", 0),
            "register_to": idx_params.get("register_to", []),
            "maps": idx_params.get("maps", {}),
            "necrosis_map": idx_params.get("necrosis_map", None),
        }

        if crop := idx_params.get("crop", None):
            obj["crop"] = crop

        if maps:
            necrosis_img = maps / (necrosis_dir_name or "necrosis") / f"{img.name}.tif"
            if necrosis_img.exists():
                obj["necrosis_map"] = LazyAntsImage(necrosis_img)

            for maps_dir in maps.iterdir():
                if not maps_dir.is_dir():
                    continue

                map_name = maps_dir.name

                map_img = maps_dir / f"{img.name}.tif"

                if not map_img.is_file():
                    print(f"No {map_name} map corresponding to {img.name} found, continuing.")
                    continue

                map_obj: HistSlicesMaps = {
                    # There should never be multiple, since that would lead to clashing filenames
                    "map_img": LazyAntsImage(map_img),
                    "combine_type": "add",
                    "necrosis_correct": False,
                }
                obj["maps"][map_name] = map_obj

        out.append(obj)

    return out


# The point of this section is to construct the parameter objects for the MRI and Histology images.
# These contain paths to the images, as well as allow customisation of the registration workflow, in order to perform
# the tests in Results
if __name__ == "__main__":
    dicom23p = build_dicom_params(Path("23R_SC2"), slices_idx=[8, 9])
    dicom23r = build_dicom_params(Path("23R_SC2"), slices_idx=[6, 7])
    dicom23s = build_dicom_params(Path("23S_SC2"), slices_idx=[6, 7])
    dicom23t = build_dicom_params(Path("23T_SC2"), slices_idx=[7, 8])
    dicom23u = build_dicom_params(Path("23U_SC2"), slices_idx=[8, 9])
    dicom23w = build_dicom_params(Path("23W_SC2"), slices_idx=[7, 8])
    dicom23x = build_dicom_params(Path("23X_SC2"), slices_idx=[7, 8])
    dicom23y = build_dicom_params(Path("23Y_SC2"), slices_idx=[7, 8])

    base_hist: HistParams = {
        "greyscale_type": "mean",
        "loc_within": False,
        "fixed_image": 2,
        "split_multiple_register_to": False,
    }  # type: ignore

    # base_hist2 = HistParamsDC()

    base_hist_slice_params = {
        0: {"register_to": "mri_1", "rotation": 110},
        1: {"register_to": "mri_1", "rotation": 110},
        2: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        3: {"register_to": "mri_2", "rotation": 110},
        4: {"register_to": "mri_2", "rotation": 110},
    }

    hist23p: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23p",
            map_dir=Path("23yqp") / "export",
            params={**base_hist_slice_params},
        ),
    }
    hist23r: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23r",
            map_dir=Path("23yqp") / "export",
            params={**base_hist_slice_params},
        ),
    }
    hist23s: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23s",
            map_dir=Path("23yqp") / "export",
            params={**base_hist_slice_params},
        ),
    }
    hist23t: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23t", map_dir=Path("23yqp") / "export", params={**base_hist_slice_params}
        ),
    }
    hist23u: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23u", map_dir=Path("23yqp") / "export", params={**base_hist_slice_params}
        ),
    }
    hist23w: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23w",
            map_dir=Path("23yqp") / "export",
            params={
                **base_hist_slice_params,
                2: {**base_hist_slice_params[2], "rotation": 115, "crop": ((0, 0), (2500, 2800))},
                4: {**base_hist_slice_params[4], "rotation": 120, "crop": ((0, 0), (2400, 2800))},
            },
        ),
    }
    hist23x: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23x",
            map_dir=Path("23yqp") / "export",
            params={
                **base_hist_slice_params,
                1: {**base_hist_slice_params[1], "rotation": 105, "crop": ((300, 300), (2500, 3100))},
                # 4: {**base_hist_slice_params[4], "crop": ((700, 700), (5000, 5000))},
            },
        ),
    }
    hist23y: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23y",
            map_dir=Path("23yqp") / "export",
            params={**base_hist_slice_params},
        ),
    }

    def reg_params(replace_dict: dict[str, Any]):
        return {
            "type_of_transform": "SyN",
            # "out_prefix": ,
            "use_initial_affine": False,
            "singleprecision": False,
            #  "mask_all_stages": True,
            "grad_step": 0.0125,
            # "reg_iterations": (30000, 200000, 100000, 50000, 10000),
            # "aff_shrink_factors": (12, 6, 4, 2, 1),
            # "aff_iterations": (500000, 400000, 300000, 200000, 100000),
            # "aff_smoothing_sigmas": (4, 3, 2, 1, 0),
            # # "verbose": True,
            # "aff_random_sampling_rate": 1.0,
            # "aff_sampling": 128,
            # "syn_sampling": 128,
            "total_sigma": 0,
            "flow_sigma": 0,
        } | replace_dict

    all_params: dict[str, AnimalParams] = {
        "23P": {"mri": dicom23p, "hist": hist23p},
        "23R": {"mri": dicom23r, "hist": hist23r},
        "23S": {"mri": dicom23s, "hist": hist23s},
        "23T": {"mri": dicom23t, "hist": hist23t},
        "23U": {"mri": dicom23u, "hist": hist23u},
        "23W": {"mri": dicom23w, "hist": hist23w},
        "23X": {"mri": dicom23x, "hist": hist23x},
        "23Y": {"mri": dicom23y, "hist": hist23y},
    }

    @func_timer
    async def main():
        tasks = []
        for animal, param in all_params.items():

            # Components
            for mode in ["h&e", "mean", {"red", "blue"}]:
                param["hist"]["greyscale_type"] = mode

                string_dict = {
                    "h&e": "H&E",
                    "mean": "RGB Mean",
                }
                if mode == {"red", "blue"}:
                    pretty_string = "RB Mean"
                else:
                    pretty_string = string_dict[mode]

                tasks.append(
                    run_registration_in_thread(
                        dicom_params=param["mri"],
                        hist_params=param["hist"],
                        reg_params=reg_params({"out_prefix": Path("components") / animal / pretty_string}),
                        run_name=f"{animal} {pretty_string}",
                        strict=False,
                    )
                )

            # Registration type
            for reg in [
                "Rigid",
                "Affine",
                "SyNOnly",
                "SyN",
                "SyNRA",
            ]:

                tasks.append(
                    run_registration_in_thread(
                        dicom_params=param["mri"],
                        hist_params=param["hist"],
                        reg_params=reg_params({"out_prefix": Path("reg_types") / animal / reg, "type_of_transform": reg}),
                        run_name=f"{animal} {reg}",
                        strict=False,
                    )
                )

            # MRI allocation
            for mri in ["mri_1", "mri_2"]:

                hist_params = deepcopy(param["hist"])
                for slices in hist_params["slices"]:
                    slices["register_to"] = mri

                tasks.append(
                    run_registration_in_thread(
                        dicom_params=param["mri"],
                        hist_params=hist_params,
                        reg_params=reg_params({"out_prefix": Path("allocation") / animal / mri}),
                        run_name=f"{animal} {mri}",
                        strict=False,
                    )
                )

        await run_many_registrations(tasks, 2)

    asyncio.run(main())
