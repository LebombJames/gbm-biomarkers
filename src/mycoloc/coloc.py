from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import gc
from functools import partial
from pathlib import Path

import ants
import numpy as np
from ants import ANTsImage
from pandas import DataFrame

from src.mycoloc.__types import *
from src.mycoloc.config import DEBUG
from src.mycoloc.hist import (
    allocate_hists,
    create_hist_volume,
    prepare_hist,
    register_hist_within,
    transform_original_hist,
)
from src.mycoloc.img_utils import create_checkerboard, prepare_mri
from src.mycoloc.LazyAntsImage import LazyAntsImage
from src.mycoloc.maps import combine_maps, process_maps
from src.mycoloc.plots import collate_checkerboard_plots, plot_roi_intensity
from src.mycoloc.utils import ensure_path_exists, func_timer, pretty_hist_filename, pretty_mri_key


def run_registration_in_thread(
    dicom_params: DicomParams, hist_params: HistParams, reg_params: dict[str, Any], run_name="coloc", strict=True
):
    return {
        "dicom_params": dicom_params,
        "hist_params": hist_params,
        "reg_params": reg_params,
        "run_name": run_name,
        "strict": strict,
    }


async def run_many_registrations(jobs: list[dict[str, Any]], max_workers: int = 3) -> list[RegPlots | None]:
    """
    Run a series of registrations simultaneously. No more than `max_tasks` may run at once. Lower this number if you experience out of memory errors.

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

        # Gather results in the exact order they were submitted
        return await asyncio.gather(*tasks)


def run_registration(
    dicom_params: DicomParams, hist_params: HistParams, reg_params: dict[str, Any], run_name="coloc", strict=True
) -> RegPlots | None:
    from src.mycoloc.utils import progress

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
    df_list: list[dict[str, Any]] = []
    failures = 0

    base_out_path = Path("out") / reg_params["out_prefix"]

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(hist_slices, hist_params, dicom_params, return_reg_dict=False)
    else:
        hist_allocation = allocate_hists(hist_slices, dicom_params)

    # progress.reset(total=sum(len(v) for v in hist_allocation.values()))
    # print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

    for mri_key, mri_dict in mri_slices.items():

        map_img = mri_dict["img"]

        # Make any (X,Y,1) images (X,Y)
        mri_zero: ANTsImage = ants.slice_image(map_img.img, axis=-1, idx=0)  # type: ignore
        mri_zero.set_direction(np.eye(2))

        progress.write(f"{run_name} | Processing {mri_key}")
        mri_prepared_dict = prepare_mri(mri_zero, base_out_path / f"{mri_key}_overview.png")
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
            mri_processed.astype("uint8").to_file(ensure_path_exists(base_out_path / f"{mri_key}-processed.png"))

        for slice_details in hist_allocation.get(mri_key, []):
            progress.write(f"{run_name} | Processing histology {slice_details['img'].path.name}")

            hist_zero = slice_details["img"].greyscale_img(hist_params["greyscale_type"])
            hist_out_path = base_out_path / slice_details["img"].path.name / mri_key

            hist_processed = prepare_hist(
                hist_zero, slice_details, mri_processed, mri_mask, center=False, resample=True
            )
            hist_img = hist_processed["img"]
            hist_mask = hist_processed["mask"]

            if DEBUG:
                hist_img.astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_hist.png"))
                (hist_mask * 255).astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_mask.png"))

            affine_init = None
            if reg_params["use_initial_affine"]:
                progress.write(f"{run_name} | Generating affine initializer")
                affine_init = [
                    ants.affine_initializer(
                        fixed_image=mri_processed,
                        moving_image=hist_img,
                        mask=mri_mask,
                        radian_fraction=1.0,
                        search_factor=360,
                    )
                ]

            progress.write(f"{run_name} | Running registration")
            try:
                registered: RegistrationDict = ants.registration(
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
                #     print(f"""
                #         {hist_img.origin=}
                #         {mri_processed.origin=}
                #         {hist_mask.origin=}
                #         {mri_mask.origin=}
                #         """)

                #     print("MRI Mask unique values:", np.unique(mri_mask.numpy()))
                #     print("Hist Mask unique values:", np.unique(hist_mask.numpy()))

                #     print(f"{registered['warpedmovout'].shape=}, {mri_processed.shape=}")
                registered["warpedmovout"].astype("uint8").to_file(ensure_path_exists(hist_out_path / f"coloc.png"))
                registered["warpedmovout"].to_file(ensure_path_exists(hist_out_path / f"coloc.nii.gz"))

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

            labels = ants.label_overlap_measures(transformed_hist_mask, mri_mask)
            labels.rename(
                columns={"UnionOverlap": "UnionOverlap (Jaccard)", "MeanOverlap": "MeanOverlap (Dice)"}, inplace=True
            )
            labels.to_csv(hist_out_path / "label_measures.csv", index=False)
            # UnionOverlap = Jaccard, MeanOverlap = Dice. TotalOrTargetOVerlap = Target overlap

            if slice_details["maps"]:

                if DEBUG:
                    (transformed_hist_mask * 255).astype("uint8").to_file(
                        ensure_path_exists(hist_out_path / f"transformed_mask.png")
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
                    progress.write(f"{run_name} | Map {map_name} computed with MI: {map_dict['mutual_info']}")
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

                    df_list.append(
                        {
                            "hist_name": pretty_hist_filename(slice_details["img"].path.name),
                            "map_name": map_name,
                            "mi": map_dict["mutual_info"],
                            "mri_key": mri_key,
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

            progress.update(1)
            progress.write("---")
            gc.collect()

    combined = combine_maps(out_maps)

    # combine_maps_integral(out_maps, dicom_params["slices"])

    map_df = DataFrame(df_list, copy=False)
    try:
        map_df.to_csv(ensure_path_exists(Path(base_out_path) / "maps.csv"), index=False)
    except PermissionError:
        progress.write(f"{Path(base_out_path) / 'maps.csv'} cannot be written, is it currently open?")

    for map_key, map_group in combined.items():
        create_hist_volume(
            map_group,
            dicom_params,
            out_path=base_out_path / map_key / "combined.nii.gz",
            interp="nearestNeighbor",
            is_map=True,
        )
        for mri_key, map_img in map_group.items():
            create_hist_volume(
                {mri_key: map_img},
                dicom_params,
                out_path=base_out_path / map_key / f"{mri_key}-{map_key}.nii.gz",
                interp="nearestNeighbor",
                is_map=True,
            )
            # mri_img.to_file(str(base_out_path / map_key / f"{mri_key}-{map_key}.nii.gz"))

    if failures:
        progress.write(f"{run_name} | Registration complete with {failures} failures!")

    gc.collect()
    collate_checkerboard_plots(plots["checkerboard"], run_name, out_path=Path())
    # collate_mri_plots(plots["mri_overview"], run_name, out_path=Path())
    # collate_transformed_originals(plots["transformed_original"], run_name, out_path=Path())
    # collate_map_plots(plots["map_overview"], run_name, out_path=Path())
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

        slices (list of ints): List indices of slices (starting at 0) to be used in the registration. E.g 7 = MRIm08.dcm
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

        necrosis_dir_name (Path), optional: The name of the necrosis directory within `maps`, relative to `maps`. Defaults to `necrosis`.

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


if __name__ == "__main__":
    dicom23r = build_dicom_params(Path("23R_SC2"), [6, 7])
    dicom23s = build_dicom_params(Path("23S_SC2"), [6, 7])
    dicom23t = build_dicom_params(Path("23T_SC2"), [7, 8])
    dicom23u = build_dicom_params(Path("23U_SC2"), [8, 9])
    dicom23w = build_dicom_params(Path("23W_SC2"), [7, 8])
    dicom23x = build_dicom_params(Path("23X_SC2"), [7, 8])
    dicom23y = build_dicom_params(Path("23Y_SC2"), [7, 8])

    base_hist: HistParams = {
        "greyscale_type": "mean",
        "loc_within": False,
        "fixed_image": 2,
        "split_multiple_register_to": False,
    }  # type: ignore

    base_hist_slice_params = {
        0: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        1: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        2: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        3: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        4: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
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
            # for mode in ["h&e", "mean", {"red", "blue"}]:
            #     param["hist"]["greyscale_type"] = mode

            #     string_dict = {
            #         "h&e": "H&E",
            #         "mean": "RGB Mean",
            #     }
            #     if mode == {"red", "blue"}:
            #         pretty_string = "RB Mean"
            #     else:
            #         pretty_string = string_dict[mode]

            tasks.append(
                run_registration_in_thread(
                    dicom_params=param["mri"],
                    hist_params=param["hist"],
                    reg_params=reg_params({"out_prefix": Path(f"{animal}_mri_1")}),
                    run_name=f"{animal}_mri_1",
                )
            )

            # for reg in [
            #     # "Rigid", "Affine", "SyNOnly",
            #     "SyN",
            #     "SyNRA",
            # ]:
            #     reg_params_dict = reg_params({"out_prefix": Path(f"{animal}_{reg}"), "type_of_transform": reg})

            #     tasks.append(
            #         run_registration_in_thread(
            #             dicom_params=param["mri"],
            #             hist_params=param["hist"],
            #             reg_params=reg_params_dict,
            #             run_name=f"{animal} {reg}",
            #             strict=False,
            #         )
            #     )

        await run_many_registrations(tasks, 2)

        tasks = [
            # run_registration(
            #     dicom_params=dicom23r,
            #     hist_params=hist23r,
            #     reg_params=reg_params({"out_prefix": Path("23R")}),
            #     run_name="23R",
            # ),
            # run_registration(
            #     dicom_params=dicom23s,
            #     hist_params=hist23s,
            #     reg_params=reg_params({"out_prefix": Path("23S")}),
            #     run_name="23S",
            # ),
            # run_registration(
            #     dicom_params=dicom23t,
            #     hist_params=hist23t,
            #     reg_params=reg_params({"out_prefix": Path("23T")}),
            #     run_name="23T",
            # ),
            # run_registration(
            #     dicom_params=dicom23u,
            #     hist_params=hist23u,
            #     reg_params=reg_params({"out_prefix": Path("23U")}),
            #     run_name="23U",
            # ),
            # run_registration(
            #     dicom_params=dicom23w,
            #     hist_params=hist23w,
            #     reg_params=reg_params({"out_prefix": Path("23W")}),
            #     run_name="23W",
            # ),
            # run_registration(
            #     dicom_params=dicom23x,
            #     hist_params=hist23x,
            #     reg_params=reg_params({"out_prefix": Path("23X")}),
            #     run_name="23X",
            # ),
            # run_registration(
            #     dicom_params=dicom23y,
            #     hist_params=hist23y,
            #     reg_params=reg_params({"out_prefix": Path("23Y")}),
            #     run_name="23Y",
            # ),
        ]

        # plot_integral_table()

        # mypprint(f"{plots=}")

        # plot_for_runs(plots, out_path=Path())

        # tasks = [
        #     run_registration_in_thread(
        #         dicom_params=dicom23x,
        #         hist_params=hist23x,
        #         reg_params=reg_params({"out_prefix": Path("23X_Rigid"), "type_of_transform": "Rigid"}),
        #         run_name="Rigid",
        #     ),
        # ]

        # await run_many_registrations(tasks, max_tasks=3)

    asyncio.run(main())

    # i = LazyAntsImage(Path("HnE") / "23r" / "240920_GCBA_23r_HnE20x_S1.svs", level=2)
    # # m = prepare_mri(LazyAntsImage(Path("23R_SC2") / "MRIm09.dcm", dimension=2).img)
    # a = i.greyscale_img("h&e")
    # b = i.greyscale_img("h")
    # c = i.greyscale_img("e")
    # d = i.greyscale_img("mean")
    # a.to_file("a.tif")
    # b.to_file("b.tif")
    # c.to_file("c.tif")
    # # d.to_file("d.tif")

    # import skimage.exposure as se

    # # out = se.equalize_hist(b)
    # out = a
    # out.to_file("histogram.tif")
    # d.astype("uint8").to_filename("d.png")
