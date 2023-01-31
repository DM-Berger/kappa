import sys
from itertools import combinations
from pathlib import Path
from typing import (Any, List, Optional, Sequence, Tuple, Union, cast,
                    no_type_check)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confusion
from tqdm import tqdm
from typing_extensions import Literal


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


def nondiag(x: ndarray) -> ndarray:
    return x[~np.eye(x.shape[0], dtype=bool)]


def gmean(x: ArrayLike) -> float:
    return float(np.exp(np.log(x).mean()))


if __name__ == "__main__":
    dfs = []
    N_CLASSES = 4
    N = 1000
    N_REP = 20
    CORR_STRENGTHS = [((1 - 0.1 * p) / 2, 0.1 * p, (1 - 0.1 * p) / 2) for p in range(11)]
    RATERS = ["same", "diff", "non-random"]
    pbar = tqdm(total=len(CORR_STRENGTHS) * len(RATERS) * len(range(2, 50)) * N_REP)

    for raters in RATERS:
        for strength, corr_strength in enumerate(CORR_STRENGTHS):
            for N_CLASSES in range(2, 50):

                CLASSES = list(range(N_CLASSES))
                # DIST = [N_CLASSES - 0.5 * n for n in range(N_CLASSES)]
                DIST_EXP = sorted(softmax(np.random.standard_exponential(N_CLASSES)))
                DIST_UNIF = sorted(softmax(np.random.uniform(0, 1, N_CLASSES)))
                DIST_STD = sorted(softmax(np.random.standard_normal(N_CLASSES)))

                DIST1 = DIST_UNIF
                DIST2 = DIST_UNIF if raters == "same" else DIST_EXP
                DIST3 = DIST_UNIF if raters == "same" else DIST_STD

                for i in range(N_REP):

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
                        y2_errs = np.clip(
                            y2_errs, a_min=np.min(CLASSES), a_max=np.max(CLASSES)
                        )
                        y3_errs = np.clip(
                            y3_errs, a_min=np.min(CLASSES), a_max=np.max(CLASSES)
                        )
                        y_errs = [y1_errs, y2_errs, y3_errs]
                        e_corr = np.mean([np.corrcoef(*comb)[0, 1] for comb in combinations(y_errs, 2)])
                        y1[::4] = y1_errs
                        y2[::4] = y2_errs
                        y3[::4] = y3_errs
                    else:
                        y1[::4] = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST1)
                        y2[::4] = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST2)
                        y3[::4] = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST3)
                        y_errs = [y1[::4], y2[::4], y3[::4]]
                        e_corr = np.mean([np.corrcoef(*comb)[0, 1] for comb in combinations(y_errs, 2)])
                    # y2[::4] = np.random.choice(CLASSES, size=len(y1[::4]), p=DIST1)
                    # e1 = y1 != y
                    # e2 = y2 != y
                    # e3 = y3 != y

                    ys = [y1, y2, y3]
                    y_combs = list(combinations(ys, 2))
                    ec_r = np.mean([np.corrcoef(*comb)[0, 1] for comb in y_combs])

                    accs = [acc(y, yy) for yy in ys]
                    amean = gmean(accs)
                    # acc_y = acc(y1, y2)

                    ec_y = np.mean([ec(y, *comb) for comb in y_combs])
                    # ec_y = ec(y, y1, y2)
                    ecl = np.mean([ec_local(y, *comb) for comb in y_combs])
                    # acc_e = acc(e1, e2)
                    # C = confusion(y1, y2)
                    # sum0 = np.sum(C, axis=0)
                    # sum1 = np.sum(C, axis=1)
                    # E = np.outer(sum0, sum1)
                    # E_norm = E / np.sum(sum0)
                    # E2 = C * C.T
                    # print("Confusion:")
                    # print(C / N)
                    # print("Expected:")
                    # print(np.round(E / N, 4))
                    # print("Expected v2:")
                    # print(np.round(E2 / N, 4))

                    # C_acc = np.sum(np.diag(C)) / np.sum(C)
                    C_1 = confusion(y, y1, labels=CLASSES)
                    C_2 = confusion(y, y2, labels=CLASSES)
                    C_3 = confusion(y, y3, labels=CLASSES)

                    C_12_acc = np.mean(
                        [np.mean(C1 == C2) for C1, C2 in combinations([C_1, C_2, C_3], 2)]
                    )

                    k_y = np.mean([kappa(*comb) for comb in y_combs])

                    # print(f"acc(y1, y2): {acc_y}")
                    # print(f"ec(y, y1, y2): {ec_y}")
                    # print(f"acc(e1, e2): {acc_e}")
                    # print(f"sum(diag(C)) / sum(C): {C_acc}")

                    # print(f"kappa(y1, y2): {k_y}")
                    # print(f"kappa(e1, e2): {k_e}")

                    dfs.append(
                        DataFrame(
                            {
                                "n_cls": N_CLASSES,
                                "raters": raters,
                                "corr_str": strength,
                                "e_corr": e_corr,
                                "acc_mean": amean,
                                # "acc_y": acc_y,
                                # "C_acc": C_acc,
                                "C_12_acc": C_12_acc,
                                # "acc_e": acc_e,
                                "ec_r": ec_r,
                                "ec_g": ec_y,
                                "ec_l": ecl,
                                "ec_g*acc": ec_y * amean,
                                "ec_g/acc": ec_y / amean,
                                "ec_l*acc": ecl * amean,
                                "ec_l/acc": ecl / amean,
                                "k_y": k_y,
                                # "k_e": k_e,
                            },
                            index=[i],
                        )
                    )
                    pbar.update()

    pbar.close()
    df_all = pd.concat(dfs, axis=0)
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
    print(df_all.groupby("raters").corr("spearman").round(3))
    corrs = df.corr("spearman")
    print(corrs.round(3))
