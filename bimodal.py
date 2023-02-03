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


if __name__ == "__main__":
    # n_modes = 3
    for dist in ["unif", "bimodal", "exp", "exp-r", "exp2", "exp2-r"]:
        fig, axes = plt.subplots(ncols=5, nrows=2, sharey=True, sharex=False)
        for n_classes, ax in zip([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], axes.flat):
            if dist == "unif":
                p = None
            if dist == "bimodal":
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
            elif dist == "exp-r":
                p = softmax(list(reversed(np.linspace(0, 1, n_classes))))
            elif dist == "exp2":
                p = softmax(np.exp(np.linspace(0, 1, n_classes)))
            elif dist == "exp2-r":
                p = softmax(list(reversed(np.exp(np.linspace(0, 1, n_classes)))))
            ax.hist(
                np.random.choice(list(range(n_classes)), size=1000, p=p), bins=n_classes
            )
            ax.set_title(f"n_classes={n_classes}, dist={dist}")
        fig.set_size_inches(w=16, h=10)
        plt.show(block=False)
    plt.show()
