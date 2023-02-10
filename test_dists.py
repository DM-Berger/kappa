import re
import sys
from argparse import Namespace
from functools import reduce
from itertools import combinations, repeat
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast, no_type_check
from warnings import filterwarnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from irrCAC.raw import CAC
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.stats.contingency import association, crosstab
from scipy.stats.distributions import uniform
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confusion
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from demo import Y_DIST_ORDER, get_p, rejection_sample

matplotlib.use("QtAgg")


def get_exp_p(n_classes: int, rng: Generator, max_scale: float = 20) -> ndarray:
    # we want 1 / s**20 = 100, so that the majority class is 100 times more likely
    # than the minority. That is
    # s**20 = (1/100) = 10**-2
    # 20 * log10(s) = log10( 10**-2) = -2
    # log10(s) = -0.1
    # s = 10**-0.1 = 0.7943282347
    pmin = 0.7943282347
    scale = rng.uniform(1 / 10, max_scale)
    p = np.linspace(pmin, 1, n_classes) ** scale
    p /= p.sum()
    p = -np.sort(-p)
    return p


def plot_samples(dist: str) -> None:
    rng = np.random.default_rng()
    fig, axes = plt.subplots(nrows=8, ncols=14)
    for ax in axes.flat:
        n_classes = int(rng.integers(3, 50, size=[1]))
        n_modes = rng.integers(0, ceil(n_classes / 5)) if "multi" in dist else None
        ps, modes = get_p(dist=dist, n_classes=n_classes, rng=rng, n_modes=n_modes)
        # ps = get_exp_p(n_classes=n_classes, rng=rng)
        classes = list(range(n_classes))
        # samples = rng.choice(classes, p=ps, size=1000)
        samples = rejection_sample(classes=classes, dist=dist, size=1000, p=ps, rng=rng)
        ax.hist(
            samples, bins=list(range(n_classes + 1)), color="black", edgecolor="white"
        )
        missing = 0
        for c in classes:
            if c not in samples:
                missing += 1
        if missing > 0:
            ax.set_title(f"Missing classes: {missing}")

    fig.suptitle(f"Dist = {dist}")
    fig.set_size_inches(w=32, h=24)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # for dist in Y_DIST_ORDER:
    # for dist in ["exp"]:
    for dist in ["balanced", "unif", "exp", "multimodal"]:
        plot_samples(dist)
