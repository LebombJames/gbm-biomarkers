from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.mycoloc.LazyAntsImage import LazyAntsImage
from src.mycoloc.utils import animal_id_from_filename, create_subplot_grid


# Clean the 'map_name' column globally before doing any stats or aggregations
def clean_name(name: str):
    if name == "cell_count_mri_width":
        return "MRI Width"
    return str(name).replace("cell_count_", "")


def custom_sort(index_values):
    # Convert index to numbers, turning 'MRI Width' into NaN
    numeric_keys = pd.to_numeric(index_values, errors="coerce")
    # Fill NaN with -1 so 'MRI Width' moves to the front
    return numeric_keys.fillna(-1)


def create_map_violins():
    """
    Create figure 13: Violin plot showing the effect of tile size on MI. Includes every map in `out`.
    """
    df_paths = Path("out").glob("**/maps.csv")
    dfs = [pd.read_csv(csv) for csv in df_paths]
    df = pd.concat(dfs, ignore_index=True)

    df["mi"] = df["mi"].astype(float)

    df["map_name"] = df["map_name"].apply(clean_name)

    df.to_csv("csvs/master_map.csv", index=False)

    # Aggregate and sort
    summary = df.groupby(["map_name"]).aggregate({"mi": ["mean", "median", "std"]})

    sorted = summary.sort_index(key=custom_sort)

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 6))

    # x_positions based on the sorted summary
    x_positions = list(range(len(sorted)))
    map_names = sorted.index

    violin_data = [abs(df[df["map_name"] == name]["mi"]) for name in map_names]
    violins = ax.violinplot(violin_data, positions=x_positions, showmeans=False, showmedians=False, showextrema=False)

    num_violins = len(map_names)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, num_violins - 1)) for i in range(num_violins)]

    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("gray")
        body.set_alpha(0.7)
        body.set_zorder(1)

    # Plot the line showing the median
    ax.plot(
        x_positions,
        abs(sorted[("mi", "median")]),
        marker="o",
        color="red",
        linewidth=2,
        markersize=8,
        zorder=2,
        label="Median",
    )

    # 3. Axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(map_names, rotation=45, ha="right", fontsize=17)

    # ax.set_ylim([0.0, 0.46])  # type: ignore
    ax.set_xlabel("Tile Size (μm)", fontsize=18)
    ax.set_ylabel("MI", fontsize=18)

    # 4. Spines and aesthetics
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    ax.legend(fontsize=15)
    fig.savefig("figs/fig_13_tile_size_violins.png", dpi=300)


def create_component_violins():
    """
    Create Figure 11: Violins showing MI, Dice, and Jaccard based on component selection
    """
    components = pd.read_csv("csvs/components.csv")

    components["mi"] = components["mi"].astype(float)

    summary = components.groupby(["component"]).aggregate(
        {"mi": ["mean", "median", "std"], "dice": ["mean", "median", "std"], "jaccard": ["mean", "median", "std"]}
    )
    sorted = summary.sort_index(key=custom_sort)

    for metric in ["mi", "dice", "jaccard"]:

        fig, ax = plt.subplots(layout="constrained", figsize=(8, 5))

        x_positions = list(range(len(sorted)))
        component_names = sorted.index

        violin_data = [abs(components[components["component"] == name][metric]) for name in component_names]
        violins = ax.violinplot(
            violin_data, positions=x_positions, showmeans=False, showmedians=False, showextrema=False
        )

        num_violins = len(component_names)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, num_violins - 1)) for i in range(num_violins)]

        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("gray")
            body.set_alpha(0.7)
            body.set_zorder(1)

        ax.plot(
            x_positions,
            abs(sorted[(metric, "median")]),
            marker="o",
            color="red",
            linewidth=2,
            markersize=8,
            zorder=2,
            label="Median",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(component_names, rotation=45, ha="right", fontsize=17)
        ax.tick_params(axis="y", labelsize=14)

        ax.set_xlabel("Components", fontsize=18)
        ax.set_ylabel(metric.upper() if metric == "mi" else metric.capitalize(), fontsize=18)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        fig.legend(fontsize=13, loc="outside right")
        fig.savefig(f"figs/fig_11_components_violin_{metric}.png", dpi=300)


def create_reg_type_violins():
    """
    Create Figure 8: Violin plots showing MI, Dice, Jaccard based on registration type
    """
    # Reg Types
    reg_types = pd.read_csv("csvs/reg_types.csv")

    reg_types["mi"] = reg_types["mi"].astype(float)

    # Aggregate and sort
    summary = reg_types.groupby(["reg_type"]).aggregate(
        {"mi": ["mean", "median", "std"], "dice": ["mean", "median", "std"], "jaccard": ["mean", "median", "std"]}
    )
    sorted = summary.sort_index(key=custom_sort)

    for metric in ["mi", "dice", "jaccard"]:
        # Violin plot of tile sizes
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(layout="constrained", figsize=(8, 5))

        # x_positions based on the sorted summary
        x_positions = list(range(len(sorted)))
        reg_type_names = sorted.index

        violin_data = [abs(reg_types[reg_types["reg_type"] == name][metric]) for name in reg_type_names]
        violins = ax.violinplot(
            violin_data, positions=x_positions, showmeans=False, showmedians=False, showextrema=False
        )

        num_violins = len(reg_type_names)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, num_violins - 1)) for i in range(num_violins)]

        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("gray")
            body.set_alpha(0.7)
            body.set_zorder(1)

        ax.plot(
            x_positions,
            abs(sorted[(metric, "median")]),
            marker="o",
            color="red",
            linewidth=2,
            markersize=8,
            zorder=2,
            label="Median",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(reg_type_names, rotation=45, ha="right", fontsize=17)

        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("Registration Type", fontsize=18)
        ax.set_ylabel(metric.upper() if metric == "mi" else metric.capitalize(), fontsize=18)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        ax.legend(fontsize=15)
        fig.savefig(f"figs/fig_8_reg_Types_{metric}.png", dpi=300)


def plot_sih_stacked_hist():
    """
    Create Figure 5: Stacked histograms of pixel intensity (using 150um maps)
    """

    map_dir = Path("23yqp") / "export" / "cell_count_150"

    maps = [path for path in map_dir.iterdir()]

    grouped: defaultdict[str, list[LazyAntsImage]] = defaultdict(list)
    for path in maps:
        animal = animal_id_from_filename(str(path))
        grouped[animal].append(LazyAntsImage(path))

    ndims = create_subplot_grid(len(grouped.keys()))
    fig, axs = plt.subplots(layout="constrained", figsize=(15, 15), nrows=ndims.nrows, ncols=ndims.ncols, squeeze=False)
    axs = axs.flat

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = ["1 slice", "2 slices", "3 slices", "4 slices", "5 slices"]

    for ax, (animal, imgs) in zip(axs, grouped.items()):
        ax = cast(Axes, ax)
        imgs_np = []

        for img in imgs:
            arr = np.ascontiguousarray(img.img.numpy(), dtype=np.float32)
            flat_arr = np.ravel(arr)

            if animal != "23P":
                is_valid = np.isfinite(flat_arr) & (flat_arr > 100)
            else:
                is_valid = np.isfinite(flat_arr) & (flat_arr > 5)

            flat_arr = flat_arr[is_valid]

            imgs_np.append(flat_arr)

        ax.hist(imgs_np, bins=40, stacked=True, color=colors, label=labels)  # type: ignore

        ax.set_title(animal, fontsize=18)
        ax.set_xlabel("Pixel intensity", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        ax.tick_params(axis="both", labelsize=18)
        # ax.legend()

    for ax in axs[len(grouped.keys()) :]:
        ax.set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=17)

    fig.savefig("figs/fig_5_maps_histogram.png", dpi=300)


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    create_map_violins()
