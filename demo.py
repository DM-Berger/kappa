import sys
from argparse import Namespace
from functools import reduce
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check
from warnings import filterwarnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from irrCAC.raw import CAC
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

# matplotlib.use("QtAgg")
ROOT = Path(__file__).resolve().parent
CORR_STRENGTHS = [((1 - 0.1 * p) / 2, 0.1 * p, (1 - 0.1 * p) / 2) for p in range(11)]
N = 1000
N_REP = 20
CORRS_OUT = ROOT / "corrs.parquet"
DF_OUT = ROOT / "metrics.parquet"


def softmax(x: ArrayLike) -> ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def acc(y1: ndarray, y2: ndarray) -> float:
    return np.mean(y1 == y2)


def cramer_v(y1: ndarray, y2: ndarray) -> float:
    ct = crosstab(y1, y2).count
    return float(association(observed=ct, correction=False))


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


def get_desc(vals: list[float]) -> tuple[float, float, float, float, float]:
    desc = Series(vals).describe(percentiles=[0.05, 0.95])
    mean = desc["mean"]
    rnge = desc["max"] - desc["min"]
    rrange = desc["95%"] - desc["5%"]
    mx = desc["max"]
    mn = desc["min"]
    return mean, rnge, rrange, mx, mn


def get_stats(
    y: ndarray,
    ys: list[ndarray],
    y_errs: list[ndarray] | None,
    err_idx: list[ndarray],
    n_classes: int,
) -> tuple[DataFrame, DataFrame]:
    es = [y != yy for yy in ys]
    y_combs = list(combinations(ys, 2))
    e_combs = list(combinations(es, 2))
    y_tab = np.stack(ys, axis=1)
    cac = CAC(DataFrame(y_tab), categories=list(range(n_classes)))

    # "b" for binary / boolean ,e.g. operates on binary error residuals
    e_corrs = [np.corrcoef(*comb)[0, 1] for comb in y_combs]
    e_vs = [cramer_v(*comb) for comb in y_combs]
    e_bvs = [cramer_v(*comb) for comb in e_combs]
    e_bcorrs = [np.corrcoef(*comb)[0, 1] for comb in e_combs]

    e_corr = np.mean(e_corrs)
    e_v = np.mean(e_vs)
    e_bv = np.mean(e_bvs)
    e_bcorr = np.mean(e_bcorrs)

    if y_errs is not None:
        # "s" for subs, e.g. the error subset
        e_subs = [y[idx] != y_err for y_err, idx in zip(y_errs, err_idx)]
        e_sub_combs = list(combinations(e_subs, 2))
        e_sbv = np.mean([cramer_v(*comb) for comb in e_sub_combs])
        e_sbcorr = np.mean([np.corrcoef(*comb)[0, 1] for comb in e_sub_combs])
    else:
        e_sbv = np.nan
        e_sbcorr = np.nan

    accs = [acc(y, yy) for yy in ys]

    ec_rs = [np.corrcoef(*comb)[0, 1] for comb in y_combs]
    ec_brs = [np.corrcoef(*comb)[0, 1] for comb in e_combs]
    ec_gs = [ec(y, *comb) for comb in y_combs]
    ecls = [ec_local(y, *comb) for comb in y_combs]
    ecus = [ec_union(y, *comb) for comb in y_combs]
    unions = [ecu * N for ecu in ecus]
    Cs = [confusion(y, yy, labels=list(range(n_classes))) for yy in ys]
    C_12_accs = [np.mean(C1 == C2) for C1, C2 in combinations(Cs, 2)]
    ks = [kappa(*comb) for comb in y_combs]
    kys = [kappa(y, yy) for yy in ys]

    ec_r = np.mean(ec_rs)
    ec_br = np.mean(ec_brs)
    ec_g = np.mean(ec_gs)
    ecl = np.mean(ecls)
    ecu = np.mean(ecus)
    ec_mx, mx = ec_max(y, ys)
    C_12_acc = np.mean(C_12_accs)

    # distributional stats
    amean, arange, arrange, _, _ = get_desc(accs)
    kmean, krange, krrange, kmax, kmin = get_desc(ks)
    _, urange, _, umax, umin = get_desc(unions)

    means = DataFrame(
        {
            "e_corr": e_corr,
            "e_bcorr": e_bcorr,
            "e_sbcorr": e_sbcorr,  # type: ignore
            "e_v": e_v,
            "e_bv": e_bv,
            "e_sbv": e_sbv,  # type: ignore
            "a_mean": amean,
            "union_max": mx,
            # "C_12_acc": C_12_acc,
            "ec_r": ec_r,
            # "ec_br": ec_br,
            "ec_g": ec_g,
            # "ec_gu": ecu,
            # "ec_max": ec_mx,
            "ec_l": ecl,
            "ec_g*acc": ec_g * amean,
            # "ec_g/acc": ec_y / amean,
            "ec_l*acc": ecl * amean,
            # "ec_l/acc": ecl / amean,
            "alpha": cac.krippendorff()["est"]["coefficient_value"],
            "ac2": cac.gwet()["est"]["coefficient_value"],
            "K": kmean,
        },
        index=[0],
    )

    corrs = DataFrame(
        {
            "a_rng": arange,
            "a_rrng": arrange,
            "k_rng": krange,
            "k_rrng": krrange,
            "k_mx": kmax,
            "k_mn": kmin,
            "u_rng": urange,
            "u_mx": umax,
            "u_mn": umin,
            "r(Ky, acc)": np.corrcoef(accs, kys)[0, 1],
            "r(K, ec_g)": np.corrcoef(ks, ec_gs)[0, 1],
            "r(K, ec_l)": np.corrcoef(ks, ecls)[0, 1],
            "r(ec_g, ec_l)": np.corrcoef(ec_gs, ecls)[0, 1],
        },
        index=[0],
    )

    return means, corrs


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
    df, corrs = get_stats(y=y, ys=ys, y_errs=y_errs, n_classes=n_classes)
    return pd.concat([extra, df], axis=1)


def compare_error_styles(args: Namespace) -> tuple[DataFrame, DataFrame]:
    """Compare ECs and Kappa-based agreement where errors are generated in two ways

    Also parameter s, max error set size.

    Two ways: error set percent overlap (number of samples they overlap on) and
    correlation between error sets. Both controlled by single parameter r in [0, 1].
    In overlap case, r is proportion of samples in max error set to be given a
    random label.
    In dependent case, r is probability that label is changed from y_true.

    Five distributions of errors: "unif" "exp", "exp2", "exp-r", "exp2-r" which
    describe how uniform the errors are, and the "-r" reverses the direction of
    the distribution so that errors are most on majority vs. minority class.



    """
    filterwarnings("ignore", category=RuntimeWarning)
    r = args.r  # value in [0, 1]
    s = args.err_size  # value in [0, 1]
    dist = args.dist
    error_style = args.errors
    n_classes = args.n_classes
    rng = np.random.default_rng(seed=args.seed)

    CLASSES = list(range(n_classes))
    dist_unif = sorted(softmax(rng.uniform(0, 1, n_classes)))
    y = rng.choice(CLASSES, size=N, p=dist_unif)  # y_true
    ys = [y.copy() for _ in range(5)]
    n_max_err = ceil(s * len(y))
    err_max_idx = rng.permutation(len(y))[:n_max_err]
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
    idxs = []  # binary errors on error set only
    if error_style == "independent":
        # In this cases, predictions are on the same subset of the max error set,
        # but predictions are made completely independently according to the
        # distribution `dist`. `r`` is error difference (high r means very
        # different error sets) because `r` controls how much of the max error
        # set errors occur on.
        err_size = ceil(r * n_max_err)
        for yy in ys:
            # ensure all errors
            idx = rng.permutation(n_max_err)[:err_size]
            y_err = rng.choice(CLASSES, size=err_size, p=p)
            yy[idx] = y_err
            y_errs.append(y_err)
            idxs.append(idx)
    elif error_style == "dependent":
        # In this case, predictions are dependent upon a base set of predictions.
        # This means error sets will be dependent (likely correlated) too.
        # Here, `r` essentially determines the independence
        pred_base = rng.choice(CLASSES, size=n_max_err, p=p)
        for yy in ys:
            pred_rand = rng.choice(CLASSES, size=n_max_err, p=p)
            # Large r means make more predictions random,  so less similar. Essentially,
            # err_idx below determines which samples are given random predictions
            # So as r -> 1, almost all samples get a prediction from pred_base,
            # i.e. all predictions become the same.

            base_idx = rng.choice([True, False], size=n_max_err, p=[1 - r, r])
            rand_idx = ~base_idx
            preds = np.concatenate([pred_base[base_idx], pred_rand[rand_idx]])
            yy[err_max_idx] = preds

            # y_err = pred_base[err_idx]
            # final_idx = err_max_idx[err_idx]
            # yy[final_idx] = y_err
            y_errs.append(yy[err_max_idx])
            idxs.append(err_max_idx)

    else:
        raise ValueError()

    extra = DataFrame(
        {
            "n_cls": n_classes,
            "dist": dist,
            "errors": error_style,
            "r": r,
            "s": s,
            "sr": s * r,
        },
        index=[0],
    )
    df, corrs = get_stats(y=y, ys=ys, y_errs=y_errs, err_idx=idxs, n_classes=n_classes)
    return pd.concat([extra, df], axis=1), corrs


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


def run_compare_styles(n_iter: int = 10000, force: bool = False) -> None:
    dfs = []
    STYLES = ["independent", "dependent"]
    DISTS = ["unif", "exp", "exp-r", "exp2", "exp2-r"]
    ss = np.random.SeedSequence()
    seeds = ss.spawn(n_iter)

    GRID = [
        Namespace(**{**{"seed": seed}, **args})
        for args, seed in zip(
            ParameterSampler(
                {
                    "n_classes": list(range(2, 50)),
                    # "n_classes": list(range(2, 5)),
                    "errors": STYLES,
                    "dist": DISTS,
                    "r": uniform(),
                    "err_size": uniform(),
                    "reps": list(range(N_REP)),
                },
                n_iter=n_iter,
            ),
            seeds,
        )
    ]

    if DF_OUT.exists() and CORRS_OUT.exists() and (not force):
        df_all = pd.read_parquet(DF_OUT)
        c_all = pd.read_parquet(CORRS_OUT)
    else:
        dfs, corrs = list(zip(*process_map(compare_error_styles, GRID, chunksize=1)))
        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        corrs_all = pd.concat(corrs, axis=0, ignore_index=True)
        c_all = pd.concat(
            [df_all.loc[:, ["n_cls", "dist", "errors", "r", "s", "sr"]], corrs_all],
            axis=1,
        )
        df_all.to_parquet(DF_OUT)
        c_all.to_parquet(CORRS_OUT)

    sbn.set_style("darkgrid")
    grid = sbn.relplot(
        data=df_all,
        x="a_mean",
        y="K",
        # col="dist",
        # row="errors",
        col="errors",
        hue="sr",
        palette="rocket",
        size="n_cls",
    )
    plt.show()

    dists = pd.get_dummies(df_all[["errors", "dist"]])
    df = pd.concat(
        [
            df_all.loc[:, "n_cls"].to_frame(),
            dists,
            df_all.drop(columns=["errors", "dist", "n_cls"]),
        ],
        axis=1,
    )
    # n_cols = len(df.columns)
    # fmt = ["0.3f" for _ in range(n_cols + 1)]
    # fmt[0] = "0.0f"
    # print(df.to_markdown(tablefmt="simple", floatfmt=fmt, index=False))
    pd.options.display.max_rows = 1000
    desc = df_all.groupby(["errors", "dist"]).describe()
    print(desc.round(3).T)
    print("Min values")
    print(df_all.groupby(["errors", "dist"]).min().T.round(3))
    print("Max values")
    print(df_all.groupby(["errors", "dist"]).max().T.round(3))
    # print("EC / acc_mean correlation:", df.ec_q.corr(df.acc_mean))
    # print("EC / Kappa_y correlation:", df.ec.corr(df.k_y))
    # print("EC / Kappa_e correlation:", df.ec.corr(df.k_e))
    # print(df_all.groupby("raters").corr("pearson").round(3))
    # corrs = df.corr("pearson")
    print("=" * 80)
    print("Correlations taking into account distributions")
    print("=" * 80)
    cg = df_all.groupby(["errors", "dist"]).corr("pearson")
    print(cg.round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))
    cols = list(filter(lambda c: "ec" in c, cg.columns))
    print("EC correlations taking into account distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3))
    cols = list(filter(lambda c: "K" in c, cg.columns))
    print("Kappa correlations taking into account distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3))

    print("=" * 80)
    print("Correlations ignoring distributions")
    print("=" * 80)
    cg = df_all.drop(columns="dist").groupby(["errors"]).corr("pearson")
    print(cg.round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))
    cols = list(filter(lambda c: "ec" in c, cg.columns))
    print("EC correlations ignoring distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3))

    cols = list(filter(lambda c: "K" in c, cg.columns))
    print("Kappa correlations ignoring distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3))


if __name__ == "__main__":
    # run_compare_raters()
    run_compare_styles(n_iter=50000, force=True)
