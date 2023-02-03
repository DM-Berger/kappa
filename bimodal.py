import re
import sys
from argparse import Namespace
from functools import reduce
from itertools import combinations
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

matplotlib.use("QtAgg")


def softmax(x: ArrayLike) -> ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def sim_manual_dists() -> None:
    for dist in ["unif", "bimodal", "exp", "exp-r", "exp2", "exp2-r"]:
        fig, axes = plt.subplots(ncols=5, nrows=2, sharey=True, sharex=False)
        for n_classes, ax in zip([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], axes.flat):
            if dist == "unif":
                p = None
            elif dist == "bimodal":
                extreme = n_classes / 2
                p = np.ones([n_classes])
                p[0] = p[-1] = extreme
                p /= p.sum()
            elif dist == "multimodal":
                n_modes = np.random.randint(0, n_classes)
                extreme = n_classes / n_modes
                p = np.ones([n_classes])
                p[:n_modes] = extreme
                p /= p.sum()
            elif dist == "exp":
                p = softmax(np.linspace(0, 1, n_classes))
                p = np.linspace(0, 1, n_classes)
                p /= p.sum()
            elif dist == "exp-r":
                p = softmax(list(reversed(np.linspace(0, 1, n_classes))))
                p = list(reversed(np.linspace(0, 1, n_classes)))
                p = np.array(p) / np.sum(p)
            elif dist == "exp2":
                p = softmax(np.exp(np.linspace(0, 1, n_classes)))
                p = np.exp(np.exp(np.exp(np.linspace(0, 1, n_classes))))
                p /= p.sum()
            elif dist == "exp2-r":
                p = softmax(list(reversed(np.exp(np.linspace(0, 1, n_classes)))))
                p = np.exp(np.array(list(reversed(np.exp(np.linspace(0, 1, n_classes))))))
                p /= p.sum()
            ax.hist(
                np.random.choice(list(range(n_classes)), size=1000, p=p), bins=n_classes
            )
            ax.set_title(f"n_classes={n_classes}, dist={dist}")
        fig.set_size_inches(w=16, h=10)
        plt.show(block=False)
    plt.show()


def sim_beta_dists() -> None:
    rng = np.random.default_rng()
    for n_classes in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
        for ax in axes.flat:
            alpha = rng.uniform(1e-5, 10)
            beta = rng.uniform(1e-5, 10)
            p = rng.beta(alpha, beta, size=n_classes)
            p /= p.sum()
            p = sorted(p)
            ax.hist(
                np.random.choice(list(range(n_classes)), size=1000, p=p),
                bins=n_classes,
                color="black",
            )
            ax.set_title(f"a={alpha:03f}, b={beta:03f}")
        fig.set_size_inches(w=16, h=10)
        fig.suptitle(f"n_classes={n_classes}")
        fig.tight_layout()
        plt.show(block=False)
    plt.show()


def sim_beta_dist(kind="bimodal") -> None:
    rng = np.random.default_rng()
    n_classes = 25
    fig, axes = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True)
    for ax in axes.flat:
        if kind == "bimodal":
            alpha = beta = rng.uniform(1e-5, 10)
            # beta = rng.uniform(1e-5, 0.5)
        else:
            alpha = rng.uniform(1e-5, 10)
            beta = rng.uniform(1e-5, 10)
        p = rng.beta(alpha, beta, size=n_classes)
        p /= p.sum()
        p = -np.sort(-p)
        ax.hist(
            np.random.choice(list(range(n_classes)), size=1000, p=p),
            bins=n_classes,
            color="black",
        )
        ax.set_title(f"a={alpha:.3f}, b={beta:.3f}")
    fig.set_size_inches(w=16, h=10)
    fig.suptitle(f"n_classes={n_classes}")
    fig.tight_layout()
    plt.show()


def sim_exp_dist() -> None:
    rng = np.random.default_rng()
    n_classes = 5
    fig, axes = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True)
    for ax in axes.flat:
        scale = rng.uniform(1 / 10, 20)
        p = np.linspace(0, 1, n_classes) ** scale
        p /= p.sum()
        p = -np.sort(-p)
        ax.hist(
            np.random.choice(list(range(n_classes)), size=1000, p=p),
            bins=n_classes,
            color="black",
        )
        ax.set_title(f"scale={scale:.3f}")
    fig.set_size_inches(w=16, h=10)
    fig.suptitle(f"n_classes={n_classes}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # sim_beta_dist(None)
    sim_exp_dist()
