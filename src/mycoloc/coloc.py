from __future__ import annotations

import asyncio
from pathlib import Path

import ants
import numpy as np
from ants import ANTsImage

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
from src.mycoloc.plots import plot_roi_intensity
from src.mycoloc.utils import ensure_path_exists, func_timer, mypprint, progress


def run_registration(dicom_params: DicomParams, hist_params: HistParams, reg_params: RegParams):
    mri_slices = dicom_params["slices"]
    hist_slices = hist_params["slices"]

    out_maps = []

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(hist_slices, hist_params, dicom_params, return_reg_dict=False)
    else:
        hist_allocation = allocate_hists(hist_slices, dicom_params)

    # progress.reset(total=sum(len(v) for v in hist_allocation.values()))
    # print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

    progress.total += sum(len(v) for v in hist_allocation.values())  # type: ignore
    progress.refresh()

    base_out_path = Path("out") / reg_params["out_prefix"]

    for mri_key, mri_dict in mri_slices.items():

        mri_img = mri_dict["img"]

        # Make any (X,Y,1) images (X,Y)
        mri_zero: ANTsImage = ants.slice_image(mri_img.img, axis=-1, idx=0)  # type: ignore
        mri_zero.set_direction(np.eye(2))

        progress.write(f"Processing {mri_key}")
        mri_prepared_dict = prepare_mri(mri_zero)
        mri_processed = mri_prepared_dict["img"]
        mri_mask = mri_prepared_dict["mask"]  # type: ignore
        if DEBUG:
            mri_processed.astype("uint8").to_file(ensure_path_exists(base_out_path / f"{mri_key}-processed.png"))

        for slice_details in hist_allocation[mri_key]:
            # progress.set_description(f"{mri_key} | {slice_details['img'].path.name}")
            progress.write(f"Processing histology {slice_details['img'].path.name}")

            hist_zero = slice_details["img"].greyscale_img(hist_params["greyscale_type"])
            hist_out_path = base_out_path / slice_details["img"].path.name / mri_key

            hist_processed = prepare_hist(
                hist_zero, slice_details, mri_processed, mri_mask, center=False, resample=True
            )
            hist_img = hist_processed["img"]
            hist_mask = hist_processed["mask"]  # type: ignore

            # Sanity check
            # hist_mask = ants.copy_image_info(hist_img, hist_mask)
            # mri_mask = ants.copy_image_info(mri_processed, mri_mask)

            if DEBUG:
                hist_img.astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_hist.png"))
                (hist_mask * 255).astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_mask.png"))

            affine_init = None
            if reg_params["use_initial_transform"]:
                progress.write("Generating affine initializer")
                affine_init = ants.affine_initializer(fixed_image=mri_processed, moving_image=hist_img)

            progress.write("Running registration")
            registered: RegistrationDict = ants.registration(
                fixed=mri_processed,
                moving=hist_img,
                mask=mri_mask,
                moving_mask=hist_mask,
                initial_transform=affine_init,
                **reg_params,
            )

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

            progress.write("Transforming the original high-res histology")
            transform_original_hist(
                hist_zero, slice_details, mri_processed, mri_mask, registered["fwdtransforms"], out_path=hist_out_path
            )

            if slice_details["maps"]:
                transformed_mask: ANTsImage = ants.apply_transforms(
                    transformlist=registered["fwdtransforms"],
                    fixed=hist_mask,
                    moving=hist_mask,
                    interpolator="genericLabel",
                )  # type: ignore

                if DEBUG:
                    (transformed_mask * 255).astype("uint8").to_file(
                        ensure_path_exists(hist_out_path / f"transformed_mask.png")
                    )

                maps = process_maps(
                    slice_details,
                    registered["fwdtransforms"],
                    mri_processed,
                    hist_mask,
                    mri_key=mri_key,
                    dicom_params=dicom_params,
                    out_path=hist_out_path,
                )

                out_maps.append(*maps.values())

                for map_name, map_dict in maps.items():
                    progress.write(f"Map {map_name} computed with MI: {map_dict['mutual_info']}")

                    plot_roi_intensity(
                        map_dict["img"],
                        mri_processed,
                        roi_hist=((40, 100), (115, 140)),
                        roi_mri=((40, 100), (115, 140)),
                        out_path=hist_out_path / "maps" / f"{map_name} overview.png",
                        mi=map_dict["mutual_info"],
                    )

            create_checkerboard(mri_processed, registered["warpedmovout"], squares=(8, 8), out_path=hist_out_path)

            create_hist_volume({mri_key: registered["warpedmovout"]}, dicom_params, out_path=hist_out_path)

            progress.update(1)
            progress.write("---")
    combined = combine_maps(out_maps)
    # mypprint(combined)
    for map_key, map_group in combined.items():
        create_hist_volume(
            map_group, dicom_params, out_path=base_out_path / map_key / "combined.nii.gz", interp="genericLabel"
        )
        for mri_key, mri_img in map_group.items():
            create_hist_volume(
                {mri_key: mri_img},
                dicom_params,
                out_path=base_out_path / map_key / f"{mri_key}-{map_key}.nii.gz",
                interp="genericLabel",
            )
            # mri_img.to_file(str(base_out_path / map_key / f"{mri_key}-{map_key}.nii.gz"))


# print(combined)


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

    volume: ANTsImage = out["volume"]  # type: ignore

    for i, slice_idx in enumerate(slices_idx, start=1):
        sliced: ANTsImage = ants.slice_image(volume, axis=-1, idx=slice_idx)  # type: ignore
        sliced_np = sliced.numpy()[..., np.newaxis]

        sliced_ants = ants.from_numpy(
            sliced_np, spacing=volume.spacing, origin=volume.origin, direction=volume.direction
        )

        out["slices"][f"mri_{i}"] = {"img": LazyAntsImage(sliced_ants), "index": slice_idx}  # type: ignore

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

        crop = idx_params.get("crop", None)
        if crop:
            obj["crop"] = crop

        if maps:
            necrosis_img = maps / (necrosis_dir_name or "necrosis") / f"{img.name}.tif"
            if necrosis_img.exists():
                obj["necrosis_map"] = LazyAntsImage(necrosis_img)

            for map_dir in maps.iterdir():
                if not map_dir.is_dir():
                    continue

                map_name = map_dir.name

                map_img = map_dir / f"{img.name}.tif"

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
    # dicom23y: DicomParams = {
    #     "volume": ants.dicom_read("23Y_SC2"),
    #     "slices": {
    #         "mri_1": {
    #             "img": LazyAntsImage(Path("23Y_SC2") / "MRIm08.dcm", dimension=3),
    #             "index": 7,
    #         },
    #         "mri_2": {
    #             "img": LazyAntsImage(Path("23Y_SC2") / "MRIm09.dcm", dimension=3),
    #             "index": 8,
    #         },
    #     },
    # }
    hist_params: HistParams = {
        "loc_within": False,
        "fixed_image": 2,
        # "use_masks": True,
        "greyscale_type": "mean",
        "slices": [
            {
                "img": LazyAntsImage(Path("HnE") / "23Y" / "240920_GCBA_23y_HnE20x_S1.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": LazyAntsImage(
                            Path("23yqp") / "export" / "cell_count" / "240920_GCBA_23y_HnE20x_S1.svs.tif"
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_1",
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("HnE") / "23Y" / "240920_GCBA_23y_HnE20x_S2.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": LazyAntsImage(
                            Path("23yqp") / "export" / "cell_count" / "240920_GCBA_23y_HnE20x_S2.svs.tif"
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_1",
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("HnE") / "23Y" / "240920_GCBA_23y_HnE20x_S3.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": LazyAntsImage(
                            Path("23yqp") / "export" / "cell_count" / "240920_GCBA_23y_HnE20x_S3.svs.tif"
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": ["mri_1", "mri_2"],
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("HnE") / "23Y" / "240920_GCBA_23y_HnE20x_S4.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": LazyAntsImage(
                            Path("23yqp") / "export" / "cell_count" / "240920_GCBA_23y_HnE20x_S4.svs.tif"
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_2",
                "necrosis_map": None,
                "crop": ((0, 3000), (2816, 7000)),
            },
            {
                "img": LazyAntsImage(Path("HnE") / "23Y" / "240920_GCBA_23y_HnE20x_S5.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": LazyAntsImage(
                            Path("23yqp") / "export" / "cell_count" / "240920_GCBA_23y_HnE20x_S5.svs.tif"
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_2",
                "necrosis_map": None,
            },
        ],
    }
    reg: RegParams = {"type_of_transform": "SyNRA", "out_prefix": Path("23Y"), "use_initial_transform": True}
    # hist_params = generate_maps_for_params(hist_params, {"cell count": {"script_name": "cell count mask", "args": []}})

    dicom23r = build_dicom_params(Path("23R_SC2"), [7, 8])
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
    }  # type: ignore
    base_hist_slice_params = {
        0: {"register_to": "mri_1", "rotation": 110},
        1: {"register_to": "mri_1", "rotation": 110},
        2: {"register_to": ["mri_1", "mri_2"], "rotation": 110},
        3: {"register_to": "mri_2", "rotation": 110},
        4: {"register_to": "mri_2", "rotation": 110},
    }
    hist23r: HistParams = {
        **base_hist,
        "slices": build_hist_slices(
            Path("HnE") / "23r",
            map_dir=Path("23yqp") / "export",
            params={
                **base_hist_slice_params,
                2: {**base_hist_slice_params[2], "rotation": 105, "crop": ((0, 3000), (2500, 8000))},
            },
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

    def reg_params(out_dir: Path) -> RegParams:
        return {"type_of_transform": "SyNRA", "out_prefix": out_dir, "use_initial_transform": False}

    @func_timer
    async def main():
        tasks = [
            asyncio.to_thread(
                run_registration, dicom_params=dicom23r, hist_params=hist23r, reg_params=reg_params(Path("23R"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23s, hist_params=hist23s, reg_params=reg_params(Path("23S"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23t, hist_params=hist23t, reg_params=reg_params(Path("23T"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23u, hist_params=hist23u, reg_params=reg_params(Path("23U"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23w, hist_params=hist23w, reg_params=reg_params(Path("23W"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23x, hist_params=hist23x, reg_params=reg_params(Path("23X"))
            ),
            asyncio.to_thread(
                run_registration, dicom_params=dicom23y, hist_params=hist23y, reg_params=reg_params(Path("23Y"))
            ),
        ]

        await asyncio.gather(*tasks)

    asyncio.run(main())

    # batch_start = time.perf_counter()
    # run_registration(dicom_params=dicom23r, hist_params=hist23r, reg_params=reg_params(Path("23R")))
    # run_registration(dicom_params=dicom23s, hist_params=hist23s, reg_params=reg_params(Path("23S")))
    # run_registration(dicom_params=dicom23t, hist_params=hist23t, reg_params=reg_params(Path("23T")))
    # run_registration(dicom_params=dicom23u, hist_params=hist23u, reg_params=reg_params(Path("23U")))
    # run_registration(dicom_params=dicom23w, hist_params=hist23w, reg_params=reg_params(Path("23W")))
    # run_registration(dicom_params=dicom23x, hist_params=hist23x, reg_params=reg_params(Path("23X")))
    # run_registration(dicom_params=dicom23y, hist_params=hist23y, reg_params=reg_params(Path("23Y")))
    # batch_end = time.perf_counter()
    # print(f"Time: {batch_end - batch_start:.2f}")

    # dummy_img: ANTsImage = ants.image_read(
    #     str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S5.svs_cellularity_100um.tif")
    # )  # type: ignore
    # test: dict[str, list[ProcessedMap]] = {
    #     "mri_1": [
    #         {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_1", "mutual_info": 0.3, "combine_type": "add"},
    #         {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_1", "mutual_info": 0.5, "combine_type": "add"},
    #     ],
    #     "mri_2": [
    #         {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_2", "mutual_info": 0.3, "combine_type": "add"},
    #         {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_2", "mutual_info": 0.5, "combine_type": "add"},
    #     ],
    # }

    # combine_maps(test)
