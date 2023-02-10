from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from demo import get_p

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True, parents=True)


def make_plot() -> None:
    SEED = 9
    N = 10000
    print(SEED)
    n_classes = 10
    classes = list(range(n_classes))
    dists = ["balanced", "unif", "multimodal", "exp"]
    titles = {
        "balanced": "Balanced",
        "unif": "Uniform",
        "multimodal": "Multi-modal",
        "exp": "Exponential",
    }
    N_ROWS, N_COLS = 3, len(dists)
    rng = np.random.default_rng(seed=SEED)

    fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS, sharex=True, sharey=True)
    for row in range(N_ROWS):
        for col, dist in enumerate(dists):
            ax: Axes = axes[row][col]
            p = get_p(
                dist=dist, n_classes=n_classes, rng=rng, n_modes=rng.choice([3, 4, 5])
            )[0]
            y = rng.choice(classes, size=N, replace=True, p=p)
            bins = ax.hist(
                y, color="black", bins=list(range(n_classes + 1)), edgecolor="white"
            )[1]
            if row == 0:
                ax.set_title(titles[dist])
                ax.set_xticks([])
            if row == N_ROWS - 1:
                ticks = bins[::2]
                labels = [str(int(tick)) for tick in ticks]
                # ticks = np.linspace(0, X_MAX, len(labels))
                ax.set_xticks(ticks, labels=labels, minor=False)
            ax.set_xlim(0, n_classes)
    fig.set_size_inches(h=5, w=8.5)
    fig.suptitle(
        rf"{N} Sampled Class Frequencies from Random $\mathcal{{Y}}$ Distributions According to Scheme"
    )
    fig.text(x=0.5, y=0.01, s="Class")
    fig.text(y=0.5, x=0.01, s="Number of samples", rotation="vertical", va="center")
    fig.tight_layout()
    fig.subplots_adjust(left=0.085, bottom=0.09)
    outfile = OUTDIR / "sample_p_distributions.png"
    fig.savefig(str(outfile), dpi=600)
    print(f"Saved plot to {outfile}")
    plt.close()


if __name__ == "__main__":
    make_plot()
