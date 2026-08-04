from collections import defaultdict
from itertools import combinations
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import pingouin as pg
from ants import ANTsImage
from pandas import DataFrame
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_ind

from scripts.figs import clean_name
from src.sihpy.__types import MRISliceDict, ProcessedMap
from src.sihpy.img_utils import prepare_mri
from src.sihpy.LazyAntsImage import LazyAntsImage
from src.sihpy.utils import animal_id_from_filename, ensure_path_exists, multi_index_to_str, pretty_hist_filename


def tile_size_anova():
    """
    Perform an ANOVA to show the effect on tile size on MI. This wasn't included in the report but is here for completeness.
    """
    df_paths = Path("out").glob("**/maps.csv")
    dfs = [pd.read_csv(csv) for csv in df_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Clean-up types
    df["mi"] = df["mi"].astype(float)
    df["map_name"] = df["map_name"].apply(clean_name)

    # Is this better when tile size is continuous?
    df_continuous = df[df["map_name"] != "MRI Width"].copy()
    df_continuous["map_name"].apply(int)
    model_cont = ols("mi ~ map_name", data=df_continuous).fit()

    print("continuous")
    print(model_cont.summary())

    # Repeated measures ANOVA
    anova_table = pg.rm_anova(data=df, dv="mi", within="map_name", subject="hist_name")
    anova_table.to_csv(ensure_path_exists(Path("csvs") / "tile_size_anova.csv"))

    # Post-hoc t-tests
    posthoc = pg.pairwise_tests(data=df, dv="mi", within="map_name", subject="hist_name", padjust="holm")
    posthoc.to_csv("csvs/tile_size_posthoc.csv")

    # TOST test
    EQUIV_MARGIN = 0.5 * df["mi"].std()  # Moderate effect size
    ALPHA = 0.05

    groups = df["map_name"].dropna().unique()
    pairs = list(combinations(groups, 2))
    tost_results = []

    for g1, g2 in pairs:
        data1 = df[df["map_name"] == g1]["mi"].dropna()
        data2 = df[df["map_name"] == g2]["mi"].dropna()

        mean_diff = data1.mean() - data2.mean()

        p_val, a, b = ttost_ind(
            x1=data1,
            x2=data2,
            low=-EQUIV_MARGIN,
            upp=EQUIV_MARGIN,
            usevar="unequal",
        )

        tost_results.append({"group1": g1, "group2": g2, "meandiff": mean_diff, "tost_pvalue_raw": p_val})

    tost_df = DataFrame(tost_results)

    # Apply Benjamini-Hochberg correction
    reject, pvals_corrected, _, _ = multipletests(pvals=tost_df["tost_pvalue_raw"], alpha=ALPHA, method="fdr_bh")

    tost_df["p-adj"] = pvals_corrected
    tost_df["equivalent"] = reject

    tost_df.to_csv(ensure_path_exists(Path("csvs") / "tile_size_tost.csv"), index=False)


def combine_maps_integral(maps: list[ProcessedMap], mri_params: dict[str, MRISliceDict]):
    """
    As `combine_maps`, but repeatedly calculates MI for SIH maps using 1 map, 2 maps, then 3 maps.
    """
    from src.sihpy.coloc import DEBUG
    from src.sihpy.maps import combine_fns

    df_rows = []

    # grouped_imgs[map_name][mri_key] = [img1, img2, ...]
    grouped_imgs: defaultdict[str, defaultdict[str, list[LazyAntsImage]]] = defaultdict(lambda: defaultdict(list))
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

    combined_out: defaultdict[str, dict[str, ANTsImage]] = defaultdict(dict)

    for map_name, mri_dict in grouped_imgs.items():
        # Get the combine type for this map name
        combine_fn = combine_fns.get(combine_types_map[map_name])

        if not combine_fn:
            raise KeyError(f"Invalid combine type: {combine_types_map[map_name]}")

        for mri_key, imgs in mri_dict.items():
            for number in range(1, 4):
                new_imgs: list[ANTsImage] = []
                for img in imgs[:number]:
                    if "middle_slice_factor" in img.flags:
                        corrected = LazyAntsImage(img.img * img.flags["middle_slice_factor"]).round()
                        new_imgs.append(corrected)
                    else:
                        new_imgs.append(img.img)

                out_img = combined_out[map_name][mri_key] = combine_fn(new_imgs)

                out_img.to_file(
                    ensure_path_exists(Path("out") / "integrals" / f"{mri_key}-{map_name}-combined-1-{number}.nii.gz")
                )

                mri: ANTsImage = ants.slice_image(mri_params[mri_key]["img"].img, axis=-1, idx=0)  # type: ignore
                mri_prepared = prepare_mri(mri)["img"]

                mi = ants.image_mutual_information(out_img, mri_prepared)

                df_rows.append(
                    {
                        "mi": mi,
                        "max_slice": number,
                        "mri_key": mri_key,
                        "animal_id": animal_id_from_filename(mri_params[mri_key]["img"].flags.get("path", "")),
                        "map_name": map_name,
                    }
                )

    out_df = pd.DataFrame(df_rows)
    out_df.to_csv(
        ensure_path_exists(
            f"csvs/integrals/{animal_id_from_filename(mri_params['mri_1']['img'].flags.get('path',''))}.csv"
        ),
        index=False,
    )

    return dict(combined_out)


def components_anova():
    """
    Perform an ANOVA into the effect of component selection on MI.
    """
    in_paths = Path("out").iterdir()
    map_df_list = []
    overlap = []

    # Find every registration that ends with a component type, read its MI, Dice, Jaccard. Perform an ANOVA with the MI
    for path in in_paths:
        split = str(path.parts[1]).split("_")
        if len(split) != 2:
            continue

        animal, component = split
        if component not in {"RGB Mean", "RB Mean", "H&E"}:
            continue

        map_df = path / "maps.csv"
        map_csv = pd.read_csv(map_df)
        map_csv["component"] = component
        map_df_list.append(map_csv)

        slices = path.glob("*S*.svs/mri_*/")
        for slice_path in slices:
            label_csv_path = slice_path / "label_measures.csv"
            csv = pd.read_csv(label_csv_path)

            overlap.append(
                {
                    "dice": csv["MeanOverlap (Dice)"][0],
                    "jaccard": csv["UnionOverlap (Jaccard)"][0],
                    "hist_name": pretty_hist_filename(slice_path),
                    "mri_key": str(label_csv_path.parent.parts[-1]),
                    "component": component,
                }
            )

    overlap_df = DataFrame(overlap)

    df = pd.concat(map_df_list)
    df: DataFrame = df[df["map_name"] == "cell_count_100"].copy()

    # Join the map df with the overlap df
    df = df.merge(overlap_df, on=["hist_name", "mri_key", "component"], how="inner")
    df.to_csv(ensure_path_exists(Path("csvs") / "components.csv"))

    df["mi"] = np.abs(df["mi"])
    grouped = df.groupby(by="component")
    summary = grouped.aggregate(
        {"mi": ["mean", "median", "std"], "dice": ["mean", "median", "std"], "jaccard": ["mean", "median", "std"]}
    )
    summary.columns = multi_index_to_str(summary.columns)
    summary: DataFrame = summary.rename(columns={"mi_mean": "MI Mean", "mi_median": "MI Median", "mi_std": "SD"})
    summary.index.names = ["Component"]

    sorted = summary.sort_values(by="MI Mean", ascending=False)
    sorted: DataFrame = sorted.round(3)

    sorted.to_csv(ensure_path_exists(Path("csvs") / "components_summary.csv"))

    # ANOVA
    anova_table = pg.rm_anova(data=df, dv="mi", within="component", subject="hist_name")
    anova_table.to_csv(ensure_path_exists(Path("csvs") / "component_anova.csv"))

    # Sphericity was violated, perform follow up tests
    spher, W, chisq, dof, pval = pg.sphericity(data=df, dv="mi", within="component", subject="hist_name")
    sphericity = DataFrame({"spher": spher, "W": W, "chisq": chisq, "dof": dof, "pval": pval}, index=[0])

    sphericity.to_csv(ensure_path_exists(Path("csvs") / "component_sphericity.csv"))

    # Post-hoc t-tests
    posthoc = pg.pairwise_tests(data=df, dv="mi", within="component", subject="hist_name", padjust="holm")
    posthoc.to_csv("csvs/component_rm_posthoc.csv")

    # TOST test
    EQUIV_MARGIN = 0.5 * df["mi"].std()
    ALPHA = 0.05

    groups = df["component"].dropna().unique()
    pairs = list(combinations(groups, 2))
    tost_results = []

    for g1, g2 in pairs:
        data1 = df[df["component"] == g1]["mi"].dropna()
        data2 = df[df["component"] == g2]["mi"].dropna()

        mean_diff = data1.mean() - data2.mean()

        p_val, a, b = ttost_ind(x1=data1, x2=data2, low=-EQUIV_MARGIN, upp=EQUIV_MARGIN, usevar="unequal")

        tost_results.append({"group1": g1, "group2": g2, "meandiff": mean_diff, "tost_pvalue_raw": p_val})

    tost_df = pd.DataFrame(tost_results)

    # Apply Benjamini-Hochberg correction
    reject, pvals_corrected, _, _ = multipletests(pvals=tost_df["tost_pvalue_raw"], alpha=ALPHA, method="fdr_bh")

    tost_df["p-adj"] = pvals_corrected
    tost_df["equivalent"] = reject

    tost_df.to_csv(ensure_path_exists(Path("csvs") / "components_tost.csv"), index=False)


def reg_type_anova():
    """
    Perform an ANOVA into the effect of registration type on MI.
    """
    in_paths = Path("out").iterdir()
    map_df_list = []
    overlap = []

    # Find every registration that ends with a registration type, read its MI, Dice, Jaccard. Perform an ANOVA with the MI
    for path in in_paths:
        split = str(path.parts[1]).split("_")
        if len(split) != 2:
            continue

        animal, reg_type = split
        if reg_type not in {"Rigid", "Affine", "SyNOnly", "SyN", "SyNRA"}:
            continue

        map_df = path / "maps.csv"
        map_csv = pd.read_csv(map_df)
        map_csv["reg_type"] = reg_type
        map_df_list.append(map_csv)

        slices = path.glob("*S*.svs/mri_*/")
        for slice_path in slices:
            label_csv_path = slice_path / "label_measures.csv"
            try:
                csv = pd.read_csv(label_csv_path)
            except FileNotFoundError:
                print(f"{label_csv_path} not found, did this registration fail?")
                continue

            overlap.append(
                {
                    "dice": csv["MeanOverlap (Dice)"][0],
                    "jaccard": csv["UnionOverlap (Jaccard)"][0],
                    "hist_name": pretty_hist_filename(slice_path),
                    "mri_key": str(label_csv_path.parent.parts[-1]),
                    "reg_type": reg_type,
                }
            )

    overlap_df = DataFrame(overlap)

    df = pd.concat(map_df_list)
    df: DataFrame = df[df["map_name"] == "cell_count_100"].copy()

    # Merge map dfs with MI and overlap df with Dice and Jaccard
    df = df.merge(overlap_df, on=["hist_name", "mri_key", "reg_type"], how="inner")

    df["mi"] = np.abs(df["mi"])
    df.to_csv(ensure_path_exists(Path("csvs") / "reg_types.csv"))
    grouped = df.groupby(by="reg_type")
    summary: DataFrame = grouped.aggregate(
        {"mi": ["mean", "median", "std"], "dice": ["mean", "median", "std"], "jaccard": ["mean", "median", "std"]}
    )
    summary.columns = multi_index_to_str(summary.columns)

    summary.rename(columns={"mi_mean": "MI Mean", "mi_median": "MI Median", "mi_std": "SD"}, inplace=True)
    summary.index.names = ["Registration Type"]

    sorted = summary.sort_values(by="MI Mean", ascending=False)
    sorted: DataFrame = sorted.round(3)

    sorted.to_csv(ensure_path_exists(Path("csvs") / "reg_types_summary.csv"))

    # ANOVA
    anova_table = pg.rm_anova(data=df, dv="mi", within="reg_type", subject="hist_name")
    anova_table.to_csv(ensure_path_exists(Path("csvs") / "reg_type_anova.csv"))

    # Sphericity test
    spher, W, chisq, dof, pval = pg.sphericity(data=df, dv="mi", within="reg_type", subject="hist_name")
    sphericity = DataFrame({"spher": spher, "W": W, "chisq": chisq, "dof": dof, "pval": pval}, index=[0])

    sphericity.to_csv(ensure_path_exists(Path("csvs") / "reg_type_sphericity.csv"))

    # Post-hoc pairwise t-tests
    posthoc = pg.pairwise_tests(data=df, dv="mi", within="reg_type", subject="hist_name", padjust="holm")
    posthoc.to_csv("csvs/reg_type_rm_posthoc.csv")

    # TOST equivalence test
    EQUIV_MARGIN = 0.5 * df["mi"].std()  # Moderate effect size
    ALPHA = 0.05

    groups = df["reg_type"].dropna().unique()
    pairs = list(combinations(groups, 2))
    tost_results = []

    for g1, g2 in pairs:
        data1 = df[df["reg_type"] == g1]["mi"].dropna()
        data2 = df[df["reg_type"] == g2]["mi"].dropna()

        mean_diff = data1.mean() - data2.mean()

        p_val, a, b = ttost_ind(x1=data1, x2=data2, low=-EQUIV_MARGIN, upp=EQUIV_MARGIN, usevar="unequal")

        tost_results.append({"group1": g1, "group2": g2, "meandiff": mean_diff, "tost_pvalue_raw": p_val})

    tost_df = pd.DataFrame(tost_results)

    # Apply Benjamini-Hochberg correction
    reject, pvals_corrected, _, _ = multipletests(pvals=tost_df["tost_pvalue_raw"], alpha=ALPHA, method="fdr_bh")

    tost_df["p-adj"] = pvals_corrected
    tost_df["equivalent"] = reject

    tost_df.to_csv(ensure_path_exists(Path("csvs") / "reg_type_tost.csv"), index=False)


def max_mi_test():
    """
    Test if solid images of a certain intensity produce notable changes in MI.
    """
    mri_img = prepare_mri(ants.slice_image(LazyAntsImage(Path("23Y_SC2") / "MRIm08.dcm").img, axis=-1, idx=0))["img"]  # type: ignore

    for n in range(0, 256):
        arr = np.full((mri_img.shape[0], mri_img.shape[1]), float(n))
        arr[0, 0] = 255.0 - n
        arr_img = ants.new_image_like(mri_img, arr)

        mi = ants.image_mutual_information(mri_img, arr_img)
        print(f"Intensity {n}: {mi:.20f}")
