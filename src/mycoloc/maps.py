from __future__ import annotations

import operator
from collections import defaultdict
from functools import reduce
from pathlib import Path

import ants
from ants import ANTsImage

from src.mycoloc.__types import *
from src.mycoloc.utils import ensure_path_exists


def process_maps(
    slice_details: HistSlicesDict,
    transform: list[str],
    mri_processed: ANTsImage,
    hist_mask: ANTsImage,
    out_path: Path,
    mri_key: str,
    dicom_params: DicomParams | None = None,
) -> dict[str, ProcessedMap]:
    from src.mycoloc.coloc import DEBUG, create_hist_volume, prepare_hist, progress

    if "maps" not in slice_details:
        raise AttributeError(f"Slice {slice_details['img'].path.name} has no associated maps.")

    ret_dict: dict[str, ProcessedMap] = {}

    for map_name, map_dict in slice_details["maps"].items():
        progress.write(f"Processing map {map_name}")
        map_processed = prepare_hist(
            map_dict["map_img"],
            slice_details,
            mri_processed,
            threshold=False,
            resample=True,
            interp="genericLabel",
            out_path=out_path / "maps" / "debug",
        )
        map_transformed: ANTsImage = ants.apply_transforms(
            transformlist=transform,
            fixed=map_processed,
            moving=map_processed,
            interpolator="genericLabel",
        )  # type: ignore

        mask_transformed: ANTsImage = ants.apply_transforms(
            transformlist=transform,
            fixed=hist_mask,
            moving=hist_mask,
            interpolator="genericLabel",
        )  # type: ignore

        map_masked = map_transformed * mask_transformed

        if (necrosis_map := slice_details["necrosis_map"]) and map_dict["necrosis_correct"]:
            necrosis_processed = prepare_hist(
                necrosis_map,
                slice_details,
                mri_processed,
                threshold=False,
                resample=True,
                interp="genericLabel",
                out_path=out_path / "maps" / "debug",
            )
            necrosis_transformed: ANTsImage = ants.apply_transforms(
                transformlist=transform,
                fixed=necrosis_processed,
                moving=necrosis_processed,
                interpolator="genericLabel",
            )  # type: ignore

            progress.write("Necrosis-correcting cell density map")
            final_map = necrosis_correct_density(map_masked, necrosis_transformed)  #

            # (final_map * 255).astype("uint8").to_file(
            #     ensure_path_exists(out_path / "maps" / f"{map_name}_map_corrected.nii.gz")
            # )
        else:
            final_map = map_masked

        if DEBUG:

            print(f"""
                {mri_processed.shape=}
                {mri_processed.spacing=}
                {mri_processed.origin=}
                {mri_processed.direction=}
                {map_transformed.shape=}
                {map_transformed.spacing=}
                {map_transformed.origin=}
                {map_transformed.direction=}
            """)

            (map_dict["map_img"] * 255).astype("uint8").to_file(
                ensure_path_exists(out_path / "maps" / f"{map_name}_map_raw.png")
            )
            (map_processed * 255).astype("uint8").to_file(
                ensure_path_exists(out_path / "maps" / f"{map_name}_map_processed.png")
            )

            (map_masked * 255).astype("uint8").to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_masked.png"))

            (map_transformed * 255).astype("uint8").to_file(
                ensure_path_exists(out_path / "maps" / f"{map_name}_map.png")
            )
            (map_transformed * 255).to_file(ensure_path_exists(out_path / "maps" / f"{map_name}_map.tif"))

        if mri_key and dicom_params:
            # Create a volume with identical shape to the original MRI volume with the map inserted in the appropriate place
            create_hist_volume(
                {mri_key: final_map},
                dicom_params,
                interp="genericLabel",
                out_path=out_path / "maps" / f"{map_name}.nii.gz",
            )

            if DEBUG:
                create_hist_volume(
                    {mri_key: map_transformed},
                    dicom_params,
                    interp="genericLabel",
                    out_path=out_path / "maps" / f"{map_name}unmasked.nii.gz",
                )

        mi_score: float = ants.image_mutual_information(mri_processed, final_map)
        # print(mi_score)

        ret_dict[map_name] = {
            "img": final_map,
            "mutual_info": mi_score,
            "mri_key": mri_key,
            "map_name": map_name,
            "combine_type": map_dict["combine_type"],
        }

    return ret_dict


def sum_imgs(*imgs: ANTsImage) -> ANTsImage:
    if not imgs:
        raise ValueError("Please provide at least one AntsImage.")
    return reduce(operator.add, imgs)


def mean_imgs(*imgs: ANTsImage) -> ANTsImage:
    if not imgs:
        raise ValueError("Please provide at least one AntsImage.")
    return ants.average_images(list(imgs), normalize=False)  # type: ignore


combine_fns = {"add": sum_imgs, "mean": mean_imgs}


def combine_maps(maps: list[ProcessedMap]):
    from src.mycoloc.coloc import DEBUG

    # grouped_imgs[map_name][mri_key] = [img1, img2, ...]
    grouped_imgs = defaultdict(lambda: defaultdict(list))
    combine_types_map = {}

    # list[ProcessedMap] -> {"map1": {"mri_1": [img1, img2], "mri_2": [img3, img4]}, "map2": {...} }
    for processed_map in maps:
        map_name = processed_map["map_name"]
        mri_key = processed_map["mri_key"]
        current_combine_type = processed_map["combine_type"]

        # Ensure maps of the same name have the same combine type
        if combine_types_map.setdefault(map_name, current_combine_type) != current_combine_type:
            raise ValueError(
                f"Conflicting combine_type for '{map_name}': "
                f"Expected '{combine_types_map[map_name]}', got '{current_combine_type}'. "
                "All maps with the same name must have the same combine type"
            )

        grouped_imgs[map_name][mri_key].append(processed_map["img"])

    combined_out = defaultdict(dict)

    for map_name, mri_dict in grouped_imgs.items():
        # Get the combine type for this map name
        combine_fn = combine_fns.get(combine_types_map[map_name], None)

        if not combine_fn:
            raise KeyError(f"Invalid combine type '{combine_types_map[map_name]}'.")

        for mri_key, imgs in mri_dict.items():

            combined_out[map_name][mri_key] = combine_fn(*imgs)

            if DEBUG:
                combined_out[map_name][mri_key].to_file(f"{mri_key}-{map_name}-combined.nii.gz")
                for i, img in enumerate(imgs, 1):
                    img.to_file(f"{mri_key}-{map_name}{i}.nii.gz")

    return dict(combined_out)


def necrosis_correct_density(tumour_density: ANTsImage, necrosis: ANTsImage) -> ANTsImage:
    """
    A simpler, more rudimentary necrosis correction. Simply add the necrosis% to the tumour%,
    and assume the total is a marker for overall tumour infiltration.

    Args:
        tumour_density: The tumour density map
        necrosis: The necrosis map

    Returns:
        ANTsImage: The necrosis-corrected tumour density map
    """
    ants.copy_image_info(tumour_density, necrosis)

    return tumour_density + necrosis
