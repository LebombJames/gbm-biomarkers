from pathlib import Path

import ants
import matplotlib

matplotlib.use('agg')

import numpy as np
from ants import ANTsImage
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.mycoloc.__types import *


def plot_roi_intensity(
    hist_registered: ANTsImage,
    mri: ANTsImage,
    roi_hist: ROI,
    roi_mri: ROI,
    mi: float,
    out_path: Path,
):
    cropped_hist = ants.crop_indices(hist_registered, roi_hist[0], roi_hist[1]).numpy()
    cropped_mri = ants.crop_indices(mri, roi_mri[0], roi_mri[1]).numpy()

    flat_hist = np.ravel(cropped_hist)
    flat_mri = np.ravel(cropped_mri)

    normalised_hist = flat_hist / flat_hist.max()
    normalised_mri = flat_mri / flat_mri.max()

    if np.unique(normalised_hist)[0] == np.inf or np.unique(normalised_hist)[0] == np.inf:
        raise ValueError("Pixel normalisation divided by zero. Double check your ROI isn't all black")

    fig = Figure(layout="constrained")

    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:-1, 0:])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    mask = (normalised_hist > 0) & (normalised_mri > 0)
    masked_hist = normalised_hist[mask]
    masked_mri = normalised_mri[mask]

    ax1.scatter(masked_hist, masked_mri, color="blue", s=2)

    ax1.set(xlabel="Hist intensity", ylabel="MRI Intensity", title=f"Hist vs MRI intensity (MI: {mi*-1})")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.imshow(cropped_hist)
    ax2.set(title="Hist Image")

    ax3.imshow(cropped_mri)
    ax3.set(title="MRI Image")

    fig.savefig(str(out_path))
    del fig
    #plt.close(fig)
    # plt.show()
