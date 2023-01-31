import sys
from argparse import Namespace
from functools import reduce
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.stats.distributions import uniform
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confusion
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

CORR_STRENGTHS = [((1 - 0.1 * p) / 2, 0.1 * p, (1 - 0.1 * p) / 2) for p in range(11)]
N = 1000
N_REP = 20


def softmax(x: ArrayLike) -> ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def acc(y1: ndarray, y2: ndarray) -> float:
    return np.mean(y1 == y2)


def ec(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    return np.mean((y1 != y) == (y2 != y))


def ec_local(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    e1 = y1 != y
    e2 = y2 != y
    inter = e1 & e2
    union = e1 | e2
    if np.all(~union):
        return 0
    return np.sum(inter) / np.sum(union)


def _union(es: list[ndarray]) -> ndarray:
    return reduce(lambda e1, e2: e1 | e2, es)


def ec_max(y: ndarray, ys: list[ndarray]) -> tuple[float, float]:
    es = [y != yy for yy in ys]
    norm = float(np.sum(_union(es)))  # size of largest error set
    ecs = [np.sum(e1 & e2) / norm for (e1, e2) in combinations(es, 2)]
    return float(np.mean(ecs)), norm / len(y)


def ec_union(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    e1 = y1 != y
    e2 = y2 != y
    union = e1 | e2
    return np.sum(union) / len(y)


def nondiag(x: ndarray) -> ndarray:
    return x[~np.eye(x.shape[0], dtype=bool)]


def gmean(x: ArrayLike) -> float:
    return float(np.exp(np.log(x).mean()))


def get_stats(
    y: ndarray, ys: list[ndarray], y_errs: list[ndarray], n_classes: int
) -> DataFrame:
    e_corr = np.mean([np.corrcoef(*comb)[0, 1] for comb in combinations(y_errs, 2)])
    # e1 = y1 != y
    # e2 = y2 != y
    # e3 = y3 != y

    y_combs = list(combinations(ys, 2))
    ec_r = np.mean([np.corrcoef(*comb)[0, 1] for comb in y_combs])
    accs = [acc(y, yy) for yy in ys]
    amean = gmean(accs)
    ec_y = np.mean([ec(y, *comb) for comb in y_combs])
    ecl = np.mean([ec_local(y, *comb) for comb in y_combs])
    ecu = np.mean([ec_union(y, *comb) for comb in y_combs])
    ec_mx, mx = ec_max(y, ys)
    Cs = [confusion(y, yy, labels=list(range(n_classes))) for yy in ys]
    C_12_acc = np.mean([np.mean(C1 == C2) for C1, C2 in combinations(Cs, 2)])
    k_y = np.mean([kappa(*comb) for comb in y_combs])

    return DataFrame(
        {
            "e_corr": e_corr,
            "acc_mean": amean,
            "union_max": mx,
            # "acc_y": acc_y,
            # "C_acc": C_acc,
            "C_12_acc": C_12_acc,
            # "acc_e": acc_e,
            "ec_r": ec_r,
            "ec_g": ec_y,
            "ec_gu": ecu,
            "ec_max": ec_mx,
            "ec_l": ecl,
            "ec_g*acc": ec_y * amean,
            "ec_g/acc": ec_y / amean,
            "ec_l*acc": ecl * amean,
            "ec_l/acc": ecl / amean,
            "k_y": k_y,
            # "k_e": k_e,
        },
        index=[0],
    )


def compare_raters(args: Namespace) -> DataFrame:
    raters = args.raters
    strength = args.strength
    n_classes = args.n_classes
    corr_strength = CORR_STRENGTHS[strength]

    # DIST = [N_CLASSES - 0.5 * n for n in range(N_CLASSES)]
    DIST_EXP = sorted(softmax(np.random.standard_exponential(n_classes)))
    DIST_UNIF = sorted(softmax(np.random.uniform(0, 1, n_classes)))
    DIST_STD = sorted(softmax(np.random.standard_normal(n_classes)))

    DIST1 = DIST_UNIF
    DIST2 = DIST_UNIF if raters == "same" else DIST_EXP
    DIST3 = DIST_UNIF if raters == "same" else DIST_STD

    CLASSES = list(range(n_classes))

    y = np.random.choice(CLASSES, size=N, p=DIST1)
    y1 = y.copy()
    y2 = y.copy()
    y3 = y.copy()

    if raters == "non-random":
        errs = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST_UNIF)
        y1_errs = errs
        y2_errs = errs + np.random.choice(
            [-1, 0, 1], p=corr_strength, size=len(errs)
        )  # correlated
        y3_errs = errs + np.random.choice(
            [-1, 0, 1], p=corr_strength, size=len(errs)
        )  # correlated
        y2_errs = np.clip(y2_errs, a_min=np.min(CLASSES), a_max=np.max(CLASSES))
        y3_errs = np.clip(y3_errs, a_min=np.min(CLASSES), a_max=np.max(CLASSES))
    else:
        y1_errs = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST1)
        y2_errs = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST2)
        y3_errs = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST3)

    y1[::4] = y1_errs
    y2[::4] = y2_errs
    y3[::4] = y3_errs
    ys = [y1, y2, y3]
    y_errs = [y1_errs, y2_errs, y3_errs]
    extra = DataFrame({"n_cls": n_classes, "raters": raters}, index=[0])
    df = get_stats(y=y, ys=ys, y_errs=y_errs, n_classes=n_classes)
    return pd.concat([extra, df], axis=1)


def compare_error_styles(args: Namespace) -> DataFrame:
    """Compare ECs and Kappa-based agreement where errors are generated in two ways

    Also parameter s, max error set size.

    Two ways: error set percent overlap (number of samples they overlap on) and
    correlation between error sets. Both controlled by single parameter r in [0, 1].
    In overlap case, r is proportion of samples in max error set to be given a
    random label.
    In perturb case, r is probability that label is changed from y_true.

    Five distributions of errors: "unif" "exp", "exp2", "exp-r", "exp2-r" which
    describe how uniform the errors are, and the "-r" reverses the direction of
    the distribution so that errors are most on majority vs. minority class.



    """
    filterwarnings("ignore", category=RuntimeWarning)
    r = args.r  # value in [0, 1]
    s = args.err_size  # value in [0, 1]
    dist = args.dist
    style = args.style
    n_classes = args.n_classes

    CLASSES = list(range(n_classes))
    dist_unif = sorted(softmax(np.random.uniform(0, 1, n_classes)))
    y = np.random.choice(CLASSES, size=N, p=dist_unif)
    ys = [y.copy() for _ in range(5)]
    n_err = ceil(s * len(y))
    err_max_idx = np.random.permutation(len(y))[:n_err]
    if dist == "unif":
        p = None
    elif dist == "exp":
        p = softmax(np.linspace(0, 1, n_classes))
    elif dist == "exp-r":
        p = softmax(list(reversed(np.linspace(0, 1, n_classes))))
    elif dist == "exp2":
        p = softmax(np.exp(np.linspace(0, 1, n_classes)))
    elif dist == "exp2-r":
        p = softmax(list(reversed(np.exp(np.linspace(0, 1, n_classes)))))
    else:
        raise ValueError()

    y_errs = []
    if style == "overlap":
        err_size = ceil(r * n_err)
        for yy in ys:
            idx = np.random.permutation(n_err)[:err_size]
            y_err = np.random.choice(CLASSES, size=err_size, p=p)
            yy[idx] = y_err
            y_errs.append(y_err)
    elif style == "perturb":
        for yy in ys:
            idx = err_max_idx
            y_err = np.random.choice(CLASSES, size=n_err, p=p)
            yy[idx] = y_err
            y_errs.append(y_err)
    else:
        raise ValueError()

    extra = DataFrame(
        {"n_cls": n_classes, "dist": dist, "style": style, "r": r, "s": s}, index=[0]
    )
    df = get_stats(y=y, ys=ys, y_errs=y_errs, n_classes=n_classes)
    return pd.concat([extra, df], axis=1)


def run_compare_raters() -> None:
    dfs = []
    RATERS = ["same", "diff", "non-random"]

    GRID = [
        Namespace(**args)
        for args in ParameterGrid(
            {
                "raters": RATERS,
                "strength": list(range(len(CORR_STRENGTHS))),
                "n_classes": list(range(2, 50)),
                "reps": list(range(N_REP)),
            }
        )
    ]
    GRID = list(
        filter(lambda g: not ((g.raters != "non-random") and (g.strength > 0)), GRID)
    )

    dfs = process_map(compare_raters, GRID, chunksize=1)

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    dists = pd.get_dummies(df_all["raters"])
    df = pd.concat(
        [
            df_all.loc[:, "n_cls"].to_frame(),
            dists,
            df_all.drop(columns=["raters", "n_cls"]),
        ],
        axis=1,
    )
    # n_cols = len(df.columns)
    # fmt = ["0.3f" for _ in range(n_cols + 1)]
    # fmt[0] = "0.0f"
    # print(df.to_markdown(tablefmt="simple", floatfmt=fmt, index=False))
    pd.options.display.max_rows = 1000
    print(df_all.groupby("raters").describe().round(3).T)
    # print("EC / acc_mean correlation:", df.ec_q.corr(df.acc_mean))
    # print("EC / Kappa_y correlation:", df.ec.corr(df.k_y))
    # print("EC / Kappa_e correlation:", df.ec.corr(df.k_e))
    # print(df_all.groupby("raters").corr("pearson").round(3))
    # corrs = df.corr("pearson")
    print(df_all.groupby("raters").corr("pearson").round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))


def run_compare_styles(n_iter: int = 10000) -> None:
    dfs = []
    STYLES = ["overlap", "perturb"]
    DISTS = ["unif", "exp", "exp-r", "exp2", "exp2-r"]
    N_ITER = n_iter

    GRID = [
        Namespace(**args)
        for args in ParameterSampler(
            {
                "n_classes": list(range(2, 50)),
                "style": STYLES,
                "dist": DISTS,
                "r": uniform(),
                "err_size": uniform(),
                "reps": list(range(N_REP)),
            },
            n_iter=n_iter,
        )
    ]

    dfs = process_map(compare_error_styles, GRID, chunksize=1)

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    dists = pd.get_dummies(df_all[["style", "dist"]])
    df = pd.concat(
        [
            df_all.loc[:, "n_cls"].to_frame(),
            dists,
            df_all.drop(columns=["style", "dist", "n_cls"]),
        ],
        axis=1,
    )
    # n_cols = len(df.columns)
    # fmt = ["0.3f" for _ in range(n_cols + 1)]
    # fmt[0] = "0.0f"
    # print(df.to_markdown(tablefmt="simple", floatfmt=fmt, index=False))
    pd.options.display.max_rows = 1000
    print(df_all.groupby(["style", "dist"]).describe().round(3).T)
    # print("EC / acc_mean correlation:", df.ec_q.corr(df.acc_mean))
    # print("EC / Kappa_y correlation:", df.ec.corr(df.k_y))
    # print("EC / Kappa_e correlation:", df.ec.corr(df.k_e))
    # print(df_all.groupby("raters").corr("pearson").round(3))
    # corrs = df.corr("pearson")
    print(df_all.groupby(["style", "dist"]).corr("pearson").round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))


if __name__ == "__main__":
    # run_compare_raters()
    run_compare_styles()
