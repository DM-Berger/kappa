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

from demo import Y_DIST_ORDER, get_p

matplotlib.use("QtAgg")

if __name__ == "__main__":

    rng = np.random.default_rng()
    # for dist in Y_DIST_ORDER:
    for dist in ["balanced", "unif"]:

        fig, axes = plt.subplots(nrows=8, ncols=14)
        for ax in axes.flat:
            n_classes = int(rng.integers(3, 50, size=[1]))
            n_modes = rng.integers(0, ceil(n_classes / 5)) if "multi" in dist else None
            ps, modes = get_p(dist=dist, n_classes=n_classes, rng=rng, n_modes=n_modes)
            samples = rng.choice(list(range(n_classes)), p=ps, size=50000)
            ax.hist(samples, bins=n_classes, color="black")

        fig.suptitle(f"Dist = {dist}")
        fig.set_size_inches(w=32, h=24)
        fig.tight_layout()
        plt.show()
