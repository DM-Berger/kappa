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


def get_step_ps(n_classes: int) -> ndarray:
    max_width = ceil(n_classes / 5)
    n_steps = np.random.randint(2, max(3, ceil(n_classes / 5) + 1))  # n_steps >= 2
    # step_diffs = list(reversed(sorted(np.random.uniform(0, 1, n_classes).tolist())))
    step_diffs = list(
        reversed(sorted(np.diff(np.random.uniform(0, 1, n_classes + 1)).tolist()))
    )
    # step_diffs = [1.0, *step_diffs]  # E.g. now is [1, 0.86, 0.7, ...]
    steps, step_widths = [], []
    for i in range(n_steps):
        wmax = min(max_width, n_classes - np.sum(step_widths))
        if wmax > 2:
            width = int(np.random.randint(2, wmax))
        else:
            width = 0
        steps.extend([step_diffs[i] for _ in range(width)])
        step_widths.append(width)

    n_remain = n_classes - len(steps)
    p_remain = np.random.uniform(0, 1, n_remain).tolist()
    ps = np.array([*p_remain, *steps])
    ps = -np.sort(-ps) / ps.sum()


if __name__ == "__main__":

    fig, axes = plt.subplots(nrows=8, ncols=14)
    for ax in axes.flat:
        n_classes = int(np.random.randint(3, 50, size=[1]))
        ps = get_step_ps(n_classes)

        # print("n_classes:", n_classes)
        # print("n_steps:", n_steps)
        # print("step_widths:", step_widths)
        # print("steps:", steps)
        # print("ps:", ps)

        samples = np.random.choice(list(range(n_classes)), p=ps, size=50000)
        ax.hist(samples, bins=n_classes, color="black")
        ax.set_title(f"step_widths: {step_widths}")
        ax2 = plt.twinx(ax)
        ax2.plot(ps, color="red")

    fig.set_size_inches(w=32, h=24)
    fig.tight_layout()
    plt.show()
    sys.exit()

    # step_widths = []
    # step_falls = []
    # # each step except the last must decline by an amount
    # # the sum of probs must be one, ans
    # for n in range(len(n_steps)):
    #     step_falls = np.random.uniform(0, 1, size=n_steps-1)

    # step_ps = []
    # for i in range(n_steps):
    #     step_p = np.random.uniform(1, 10)  #
    #     step_ps.append(step_p)

    # steps = {  }
    # step_falls = np.random.uniform(0, 1, size=n_steps)
    # p = np.ones(n_classes)

    # for cls_idx in range(n_classes):
    #     p[start] =
