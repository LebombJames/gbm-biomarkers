import gc
import re
from math import ceil, sqrt
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pandas as pd
from ants import ANTsImage
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.mycoloc.__types import *
from src.mycoloc.utils import animal_id_from_filename, ensure_path_exists, n_subplots, pretty_hist_filename, pretty_mri_key

matplotlib.use("agg")


def plot_roi_intensity(
    map_registered: ANTsImage,
    mri: ANTsImage,
    roi: ROI,
    mi: float,
    out_path: Path,
    hist_title: str | None = "Map",
    mri_title: str | None = "T2",
    title: str | None = "",
):
    cropped_hist = map_registered.numpy()  # ants.crop_indices(map_registered, roi[0], roi[1]).numpy()
    cropped_mri = mri.numpy()  # ants.crop_indices(mri, roi[0], roi[1]).numpy()

    flat_map = np.ravel(cropped_hist)
    flat_mri = np.ravel(cropped_mri)

    if flat_map.max() in [0.0, 0] or flat_mri.max() in [0.0, 0]:
        raise ValueError("Pixel normalisation divided by zero. Double check your ROI isn't all black")

    normalised_map = flat_map / flat_map.max()
    normalised_mri = flat_mri / flat_mri.max()

    fig = Figure(layout="constrained", figsize=(8, 8))

    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:-1, 0:])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    mask = (normalised_map > 0) & (normalised_mri > 0)
    masked_hist: npt.NDArray = normalised_map[mask]
    masked_mri: npt.NDArray = normalised_mri[mask]

    ax1.scatter(x=masked_hist, y=masked_mri, color="blue", s=2)

    # brain_mask = normalised_map > 0

    # # NEW: Background mask where Map is 0 AND MRI is > 0
    # background_mask = (normalised_map == 0) & (normalised_mri > 0)

    # # 2. Extract data for the valid brain region (Blue points)
    # x_brain = normalised_map[brain_mask]
    # y_brain = normalised_mri[brain_mask]

    # # 3. Extract data for the refined background region (Red points)
    # x_background = normalised_map[background_mask]
    # y_background = normalised_mri[background_mask]

    # # 4. Plot both sets on the same axis
    # # Plot the main brain data
    # ax1.scatter(x_brain, y_brain, color='blue', s=1, label='Brain (Map > 0)')

    # # Plot the filtered background data on top
    # ax1.scatter(x_background, y_background, color='red', s=1, label='Map = 0 & MRI > 0')

    ax1.set_title(f"{title + ' ' if title else ''} (MI: {round(mi*-1, 3)})", fontsize=22)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax1.tick_params(axis="both", labelsize=22)

    ax1.set_xlabel(f"{hist_title} Intensity", fontsize=22)
    ax1.set_ylabel(f"{mri_title} Intensity", fontsize=22)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.imshow(cropped_hist.T)
    ax2.set_title(f"{hist_title} Image", fontsize=22)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(cropped_mri.T)
    ax3.set_title(f"{mri_title} Image", fontsize=22)
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.savefig(ensure_path_exists(out_path), dpi=300)
    return fig
    # plt.close(fig)
    # plt.show()


def plot_mri_before_after(
    pairs: dict[str, tuple[ANTsImage, ANTsImage]], out_path: Path | None = None, animal: str = ""
) -> Figure:

    fig = Figure(layout="constrained", figsize=(8, 8))

    fig.suptitle(f"MRI Processing{f' ({animal})' if animal else ''}", fontsize=16, fontweight="bold")

    subfigs = fig.subfigures(nrows=len(pairs.keys()), ncols=1)

    for subfig, (mri_key, (before, after)) in zip(subfigs, pairs.items()):

        subfig.suptitle(pretty_mri_key(mri_key), fontsize=14)

        axs = subfig.subplots(nrows=1, ncols=2)

        axs[0].imshow(before.view().T, cmap="gray")
        axs[0].set_title("Before")
        axs[0].set_axis_off()

        axs[1].imshow(after.view().T, cmap="gray")
        axs[1].set_title("After")
        axs[1].set_axis_off()

    if out_path:
        fig.savefig(ensure_path_exists(out_path), dpi=300)

    return fig


def plot_for_runs(runs: list[RegPlots], out_path: Path | None = None):

    for i, run in enumerate(runs):

        # CHECKERBOARD
        ch_fig = Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(ch_fig)
        ch_fig.suptitle(f"Checkerboards ({run['name']})", fontsize=16, fontweight="bold")

        axes = ch_fig.subplots(nrows=2, ncols=2)
        if isinstance(axes, np.ndarray):
            axes = axes.flat

        for ax, checkerboard in zip(axes, run["checkerboard"]):
            ax.imshow(cast(ANTsImage, checkerboard["img"]).numpy().T, cmap="gray")
            ax.axis("off")
            ax.set(
                title=f"{pretty_hist_filename(checkerboard['hist_name'] or '')} ({pretty_mri_key(checkerboard['mri_key'])})"
            )

        for ax in axes[len(run["checkerboard"]) :]:
            ax.set_visible(False)

        ch_fig.tight_layout()

        if out_path:
            ch_fig.savefig(ensure_path_exists("plots" / out_path / run["name"] / "checkerboard.png"), dpi=300)
        del ch_fig

        # MRI
        mri_dict = {}
        for mri in run["mri_overview"]:
            mri_dict[mri["mri_key"]] = (mri["img"]["before"], mri["img"]["after"])  # type: ignore

        mri_fig = plot_mri_before_after(
            mri_dict,
            # Technically correct but feels bad
            animal=animal_id_from_filename(run["mri_overview"][0].get("animal_name", "")),
        )

        if out_path:
            mri_fig.savefig(ensure_path_exists("plots" / out_path / run["name"] / "mri_overview.png"), dpi=300)
        del mri_fig

        # TRANSFORMED ORIGINALS
        t_fig = Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(t_fig)
        t_fig.suptitle(f"Transformed histology ({run['name']})", fontsize=16, fontweight="bold")

        axes = t_fig.subplots(nrows=2, ncols=2)
        if isinstance(axes, np.ndarray):
            axes = axes.flat
        for ax, transformed in zip(axes, run["transformed_original"]):
            ax.imshow(cast(ANTsImage, transformed["img"]).numpy().T, cmap="gray")
            ax.axis("off")
            ax.set(
                title=f"{pretty_hist_filename(transformed['hist_name'] or '')} ({pretty_mri_key(transformed['mri_key'])})"
            )

        for ax in axes[len(run["transformed_original"]) :]:
            ax.set_visible(False)

        t_fig.tight_layout()

        if out_path:
            t_fig.savefig(ensure_path_exists("plots" / out_path / run["name"] / "transformed.png"), dpi=300)
        del t_fig

        # MAPS
        map_fig = Figure(figsize=(8, 7))
        map_fig.suptitle(f"Cell count maps ({run['name']})", fontsize=16, fontweight="bold")

        axes = map_fig.subplots(nrows=2, ncols=2)
        if isinstance(axes, np.ndarray):
            axes = axes.flat

        for ax, map_reg in zip(axes, run["map_overview"]):

            fig = cast(Figure, map_reg["img"])

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()

            img_array = np.asarray(rgba_buffer)
            ax.imshow(img_array)
            ax.axis("off")

        for ax in axes[len(run["map_overview"]) :]:
            ax.set_visible(False)

        map_fig.tight_layout(h_pad=0.4)

        if out_path:
            map_fig.savefig(ensure_path_exists("plots" / out_path / run["name"] / "maps.png"), dpi=300)
        del map_fig

    gc.collect()


def collate_checkerboard_plots(ch_figs: list[RegPlot], name: str, out_path: Path | None = None):
    ch_fig = Figure(figsize=(6, 9))
    # canvas = FigureCanvasAgg(ch_fig)
    ch_fig.suptitle(f"Checkerboards ({name})", fontsize=16, fontweight="bold")

    axes = ch_fig.subplots(nrows=3, ncols=2)
    if isinstance(axes, np.ndarray):
        axes = axes.flat

    for ax, checkerboard in zip(axes, ch_figs):
        ax.imshow(cast(ANTsImage, checkerboard["img"]).view().T, cmap="gray")
        ax.axis("off")
        ax.set(
            title=f"{pretty_hist_filename(checkerboard['hist_name'] or '')} ({pretty_mri_key(checkerboard['mri_key'])})"
        )

    for ax in axes[len(ch_figs) :]:
        ax.set_visible(False)

    ch_fig.tight_layout(h_pad=0.5, w_pad=0.04)

    if out_path:
        ch_fig.savefig(ensure_path_exists("plots" / out_path / name / "checkerboard.png"), dpi=300)
    return ch_fig


def collate_mri_plots(mri_figs: list[RegPlot], name: str, out_path: Path | None = None):
    mri_dict = {}
    for mri in mri_figs:
        mri_dict[mri["mri_key"]] = (mri["img"]["before"], mri["img"]["after"])  # type: ignore

    mri_fig = plot_mri_before_after(
        mri_dict,
        # Technically correct but feels bad
        animal=animal_id_from_filename(mri_figs[0].get("animal_name", "")),
    )

    if out_path:
        mri_fig.savefig(ensure_path_exists("plots" / out_path / name / "mri_overview.png"), dpi=300)
    return mri_fig


def sort_by_tile_size(data: list[RegPlot]) -> list[RegPlot]:
    def sort_key(d):
        size = re.match("cell_count_(.+)", d["map_name"])
        if size:
            size = size.group(1)
        else:
            return (2, d["map_name"])

        if size.startswith("mri"):
            return (1, 0)
        else:
            return (0, int(size))

    return sorted(data, key=sort_key)


def collate_map_plots(map_figs: list[RegPlot], name: str, out_path: Path | None = None):
    map_fig = Figure(figsize=(8, 5))
    map_fig.suptitle(f"Cell count maps ({name})", fontsize=16, fontweight="bold")

    sorted_figs = sort_by_tile_size(map_figs)

    ndims = n_subplots(len(sorted_figs))
    axes = map_fig.subplots(nrows=ndims.nrows, ncols=ndims.ncols)
    if isinstance(axes, np.ndarray):
        axes = axes.flat

    for ax, map_reg in zip(axes, sorted_figs):

        fig = cast(Figure, map_reg["img"])

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()

        img_array = np.asarray(rgba_buffer)
        ax.imshow(img_array)
        ax.axis("off")
        if "map_name" in map_reg:
            size = re.match("cell_count_(.+)", map_reg["map_name"])
            if size:
                size = size.group(1)
            if size == "mri_width":
                size = "MRI Width"
            else:
                size = f"{size}μm"
            # ax.set_title(f"Tile Size: {size}", fontsize=7)

    for ax in axes[len(sorted_figs) :]:
        ax.set_visible(False)

    map_fig.tight_layout(h_pad=0.2)

    if out_path:
        map_fig.savefig(ensure_path_exists("plots" / out_path / name / "maps.png"), dpi=300)
    return map_fig


def collate_transformed_originals(figs: list[RegPlot], name: str, out_path: Path | None = None):
    t_fig = Figure(figsize=(6, 9))
    # canvas = FigureCanvasAgg(t_fig)
    t_fig.suptitle(f"Transformed histology ({name})", fontsize=16, fontweight="bold")

    axes = t_fig.subplots(nrows=3, ncols=2)
    if isinstance(axes, np.ndarray):
        axes = axes.flat
    for ax, transformed in zip(axes, figs):
        ax.imshow(cast(ANTsImage, transformed["img"]).view().T, cmap="gray")
        ax.axis("off")
        ax.set(
            title=f"{pretty_hist_filename(transformed['hist_name'] or '')} ({pretty_mri_key(transformed['mri_key'])})"
        )

    for ax in axes[len(figs) :]:
        ax.set_visible(False)

    t_fig.tight_layout(w_pad=0.04)

    if out_path:
        t_fig.savefig(ensure_path_exists("plots" / out_path / name / "transformed.png"), dpi=300)
    return t_fig


def plot_integral_table():
    csvs = [path for path in (Path("out") / "integrals").iterdir() if path.suffix == ".csv"]
    dfs = [pd.read_csv(csv) for csv in csvs]
    df = pd.concat(dfs)

    cell_count_100 = df[df["map_name"] == "cell_count_100"].copy()
    cell_count_100["mi"] = np.abs(cell_count_100["mi"])
    integral_groups = cell_count_100.groupby(by="max_slice")
    integral_summary = integral_groups.aggregate({"mi": ["mean", "std", "median"]})

    x_positions = ["1 Slice", "2 Slices", "3 Slices"]

    fig = Figure((10, 10))
    ax = fig.subplots(nrows=1, ncols=1, squeeze=True)

    ax.errorbar(
        x=x_positions,
        y=integral_summary[("mi", "mean")],
        yerr=integral_summary[("mi", "std")],
        fmt="o-",
        capsize=4,
        color="blue",
        ecolor="black",
    )

    ax.set_xticks(range(len(x_positions)))
    ax.set_xticklabels(x_positions)

    ax.set_xlabel("Number of cell count maps")
    ax.set_ylabel("Mean MI")

    fig.savefig("integralsummary.png", dpi=300)
    integral_summary.to_csv(ensure_path_exists(Path("csvs") / "integralsummary.csv"))


if __name__ == "__main__":
    plot_integral_table()
