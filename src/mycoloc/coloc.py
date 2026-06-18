from __future__ import annotations

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
from src.mycoloc.utils import ensure_path_exists, progress


def run_registration(dicom_params: DicomParams, hist_params: HistParams, reg_params: RegParams):
    mri_slices = dicom_params["slices"]
    hist_slices = hist_params["slices"]

    out_maps = []

    if hist_params["loc_within"]:
        hist_allocation = register_hist_within(hist_slices, hist_params, dicom_params, return_reg_dict=False)
    else:
        hist_allocation = allocate_hists(hist_slices, dicom_params)

    progress.reset(total=sum(len(v) for v in hist_allocation.values()))

    # print({key: [v["img"].path.name for v in val] for key, val in hist_allocation.items()})

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
            progress.set_description(f"{mri_key} | {slice_details['img'].path.name}")
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
                hist_img.astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_hist.png"))
                (hist_mask * 255).astype("uint8").to_file(ensure_path_exists(hist_out_path / f"final_mask.png"))

                print(f"""
                    {hist_img.origin=}
                    {mri_processed.origin=}
                    {hist_mask.origin=}
                    {mri_mask.origin=}
                    """)

                print("MRI Mask unique values:", np.unique(mri_mask.numpy()))
                print("Hist Mask unique values:", np.unique(hist_mask.numpy()))

                print(f"{registered['warpedmovout'].shape=}, {mri_processed.shape=}")
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


# print(combined)


def generate_maps_for_params(hist_params: HistParams, scripts: dict[str, ScriptDict]) -> HistParams:
    """
    Run qupath scripts for each slice in the hist params

    Args:
        hist_params (HistParams): The hist params to run scripts for
        scripts (dict[str, str]): Key: the name of the resulting map (ie the key in the return value). Value: the name of the qupath script (see LazyAntsImage.run_qupath_script)

    Returns:
        HistParams: The updated hist params with the added maps
    """

    for slice_entry in hist_params["slices"]:

        for map_name, script_dict in scripts.items():
            map_output = slice_entry["img"].run_qupath_script(
                project_path=Path("23yqp"), script_name=script_dict["script_name"], script_args=script_dict["args"]
            )
            slice_entry["maps"][map_name]["map_img"] = map_output

    return hist_params


if __name__ == "__main__":
    dicom: DicomParams = {
        "volume": ants.dicom_read("23Y_SC2"),
        "slices": {
            "mri_1": {
                "img": LazyAntsImage(Path("23Y_SC2") / "MRIm08.dcm", dimension=3),
                "index": 7,
            },
            "mri_2": {
                "img": LazyAntsImage(Path("23Y_SC2") / "MRIm09.dcm", dimension=3),
                "index": 8,
            },
        },
    }
    hist_params: HistParams = {
        "loc_within": False,
        "fixed_image": 2,
        # "use_masks": True,
        "greyscale_type": "mean",
        "slices": [
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S1.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": ants.image_read(
                            str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S1.svs_cellularity_100um.tif")
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_1",
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S2.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": ants.image_read(
                            str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S2.svs_cellularity_100um.tif")
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_1",
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S3.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": ants.image_read(
                            str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S3.svs_cellularity_100um.tif")
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": ["mri_1", "mri_2"],
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S4.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": ants.image_read(
                            str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S4.svs_cellularity_100um.tif")
                        ),
                        "necrosis_correct": False,
                        "combine_type": "add",
                    },
                },
                "register_to": "mri_2",
                "necrosis_map": None,
            },
            {
                "img": LazyAntsImage(Path("23Y") / "240920_GCBA_23y_HnE20x_S5.svs"),
                "rotation": 110,
                "maps": {
                    "cell count 100": {
                        "map_img": ants.image_read(
                            str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S5.svs_cellularity_100um.tif")
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

    run_registration(dicom_params=dicom, hist_params=hist_params, reg_params=reg)

    dummy_img: ANTsImage = ants.image_read(
        str(Path("23yqp") / "export" / "240920_GCBA_23y_HnE20x_S5.svs_cellularity_100um.tif")
    )  # type: ignore
    test: dict[str, list[ProcessedMap]] = {
        "mri_1": [
            {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_1", "mutual_info": 0.3, "combine_type": "add"},
            {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_1", "mutual_info": 0.5, "combine_type": "add"},
        ],
        "mri_2": [
            {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_2", "mutual_info": 0.3, "combine_type": "add"},
            {"img": dummy_img, "map_name": "cell_count", "mri_key": "mri_2", "mutual_info": 0.5, "combine_type": "add"},
        ],
    }

    # combine_maps(test)
