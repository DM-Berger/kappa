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

# matplotlib.use("QtAgg")


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


ROOT = Path(__file__).resolve().parent
PLOTS = ensure_dir(ROOT / "plots")

Y_DIST_ORDER = [
    # "flat",  # flat and balanced never look any different
    "balanced",
    "unif",  # unif and balanced / flat DO look different
    "multimodal",  # multi-modal and step never look any different
    "exp",  # generally has most extreme behaviour
    # "step",
    # "bimodal",
]
E_DIST_ORDER = [
    # "flat",
    "balanced",  # balanced vs. balanced-r never differ either
    # "balanced-r",
    "unif",  # unif vs unif-r almost never differ visibly
    # "unif-r",
    "multimodal",  # multi-modal vs. multi-r differ for EC (corr), others
    "multimodal-r",
    "exp",  # exp vs. exp-r often differ dramatically
    "exp-r",
    # "step",
    # "step-r",
    # "bimodal",
    # "bimodal-r",
]

METRICS = METRIC_ORDER = ["pa", "ec_g", "ec_gi", "ec_l", "e_corr", "K", "e_v"]
DEPENDENCES = DEPENDENCE_ORDER = ["independent", "dependent"]
RENAMES = {
    "pa": "Percent Agreement",
    "a_mean": "Mean Accuracy",
    "ec_g": "EC (acc)",
    "ec_gi": "EC (global)",
    "ec_l": "EC (local)",
    "e_corr": "EC (corr)",
    "K": "EA (kappa)",
    "e_v": "EA (Cramer's V)",
    # "alpha": "Krippendorf's alpha",
    "s": "Error set max size",
    "r": "Error independence",
    "n_cls": "Number of classes",
    "mean_cls": "Avg. Class Prob",
}

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


def eci(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    e1 = y1 != y
    e2 = y2 != y
    inter = e1 & e2
    return float(np.sum(inter) / len(y))


def ec_local(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    e1 = y1 != y
    e2 = y2 != y
    inter = e1 & e2
    union = e1 | e2
    if np.all(~union):
        return 0
    return np.sum(inter) / np.sum(union)


def _union(es: List[ndarray]) -> ndarray:
    return reduce(lambda e1, e2: e1 | e2, es)


def ec_max(y: ndarray, ys: List[ndarray]) -> Tuple[float, float]:
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


def get_desc(vals: List[float]) -> Tuple[float, float, float, float, float]:
    desc = Series(vals).describe(percentiles=[0.05, 0.95])
    mean = desc["mean"]
    rnge = desc["max"] - desc["min"]
    rrange = desc["95%"] - desc["5%"]
    mx = desc["max"]
    mn = desc["min"]
    return mean, rnge, rrange, mx, mn


def get_stats(
    y: ndarray,
    ys: List[ndarray],
    y_errs: Optional[List[ndarray]],
    err_idx: List[ndarray],
    n_classes: int,
) -> Tuple[DataFrame, DataFrame]:
    es = [y != yy for yy in ys]
    y_combs = list(combinations(ys, 2))
    e_combs = list(combinations(es, 2))
    # y_tab = np.stack(ys, axis=1)
    # cac = CAC(DataFrame(y_tab), categories=list(range(n_classes)))

    # "b" for binary / boolean ,e.g. operates on binary error residuals
    pas = [acc(*comb) for comb in y_combs]  # Percent Agreement
    e_corrs = [np.corrcoef(*comb)[0, 1] for comb in y_combs]
    e_vs = [cramer_v(*comb) for comb in y_combs]
    e_bvs = [cramer_v(*comb) for comb in e_combs]
    e_bcorrs = [np.corrcoef(*comb)[0, 1] for comb in e_combs]

    e_corr = np.mean(e_corrs)
    e_v = np.mean(e_vs)
    # e_bv = np.mean(e_bvs)
    e_bcorr = np.mean(e_bcorrs)

    if y_errs is not None:
        # "s" for subs, e.g. the error subset
        e_subs = [y[idx] != y_err for y_err, idx in zip(y_errs, err_idx)]
        e_sub_combs = list(combinations(e_subs, 2))
        # e_sbv = np.mean([cramer_v(*comb) for comb in e_sub_combs])
        e_sbcorr = np.mean([np.corrcoef(*comb)[0, 1] for comb in e_sub_combs])
    else:
        # e_sbv = np.nan
        e_sbcorr = np.nan

    accs = [acc(y, yy) for yy in ys]

    ec_rs = [np.corrcoef(*comb)[0, 1] for comb in y_combs]
    # ec_brs = [np.corrcoef(*comb)[0, 1] for comb in e_combs]
    ec_gs = [ec(y, *comb) for comb in y_combs]
    ec_gis = [eci(y, *comb) for comb in y_combs]
    ecls = [ec_local(y, *comb) for comb in y_combs]
    ecus = [ec_union(y, *comb) for comb in y_combs]
    unions = [ecu * N for ecu in ecus]
    Cs = [confusion(y, yy, labels=list(range(n_classes))) for yy in ys]
    # C_12_accs = [np.mean(C1 == C2) for C1, C2 in combinations(Cs, 2)]
    ks = [kappa(*comb) for comb in y_combs]
    kes = [kappa(*comb) for comb in e_combs]
    kys = [kappa(y, yy) for yy in ys]

    ec_r = np.mean(ec_rs)
    # ec_br = np.mean(ec_brs)
    ec_g = np.mean(ec_gs)
    ecl = np.mean(ecls)
    # ecu = np.mean(ecus)
    # ec_mx, mx = ec_max(y, ys)
    # C_12_acc = np.mean(C_12_accs)

    # distributional stats
    amean, arange, arrange, amax, amin = get_desc(accs)
    pa_mean, pa_range, pa_rrange, pa_max, pa_min = get_desc(pas)
    kmean, krange, krrange, kmax, kmin = get_desc(ks)
    ke_mean, ke_range, ke_rrange, ke_max, ke_min = get_desc(kes)
    ecg_mean, ecg_range, ecg_rrange, ecg_max, ecg_min = get_desc(ec_gs)
    ecgi_mean, ecgi_range, ecgi_rrange, ecgi_max, ecgi_min = get_desc(ec_gis)
    ecl_mean, ecl_range, ecl_rrange, ecl_max, ecl_min = get_desc(ecls)
    umean, urange, _, umax, umin = get_desc(unions)

    cls_counts = np.unique(y, return_counts=True)[1]

    means = DataFrame(
        {
            "pa": pa_mean,
            "a_mean": amean,
            "e_corr": e_corr,
            "e_bcorr": e_bcorr,
            "e_sbcorr": e_sbcorr,  # type: ignore
            "e_v": e_v,
            # "e_bv": e_bv,
            # "e_sbv": e_sbv,  # type: ignore
            "union_max": umax,
            "max_cls": cls_counts.max() / len(y),
            "min_cls": cls_counts.min() / len(y),
            "mean_cls": cls_counts.mean() / len(y),
            # "C_12_acc": C_12_acc,
            "ec_r": ec_r,
            # "ec_br": ec_br,
            "ec_g": ec_g,
            "ec_gi": ecgi_mean,
            # "ec_gu": ecu,
            # "ec_max": ec_mx,
            "ec_l": ecl,
            # "ec_g*acc": ec_g * amean,
            # "ec_g/acc": ec_y / amean,
            # "ec_l*acc": ecl * amean,
            # "ec_l/acc": ecl / amean,
            # "alpha": cac.krippendorff()["est"]["coefficient_value"],
            # "ac2": cac.gwet()["est"]["coefficient_value"],
            "K": kmean,
            "K_e": ke_mean,
        },
        index=[0],
    )

    metrics = pd.DataFrame(
        {
            "PA": pas,
            "K": ks,
            "V": e_vs,
            "EC_g": ec_gs,
            "EC_gi": ec_gis,
            "EC_l": ecls,
        }
    )
    # get upper triangle of correlations
    cs = metrics.corr()
    idx = np.triu(np.ones_like(cs, dtype=bool), k=1).ravel()
    cs = (
        cs.stack(dropna=False)[idx]
        .reset_index()
        .rename(columns={"level_0": "m1", "level_1": "m2", "0": "r", 0: "r"})
        .sort_values(by=["m1", "m2"])
    )
    names = []
    for i in range(len(cs)):
        names.append(f"r({cs.iloc[i, 0]}, {cs.iloc[i, 1]})")
    cs["corr"] = names
    cs = cs.loc[:, ["corr", "r"]]
    cs.index = cs["corr"]
    cs.drop(columns="corr", inplace=True)

    stats = {
        "a_rng": arange,
        "a_rrng": arrange,
        "a_max": amax,
        "a_min": amin,
        "pa_rng": pa_range,
        "pa_rrng": pa_rrange,
        "pa_max": pa_max,
        "pa_min": pa_min,
        "k_rng": krange,
        "k_rrng": krrange,
        "k_max": kmax,
        "k_man": kmin,
        "ke_rng": ke_range,
        "ke_rrng": ke_rrange,
        "ke_max": ke_max,
        "ke_man": ke_min,
        "ecg_rng": ecg_range,
        "ecg_rrng": ecg_rrange,
        "ecg_max": ecg_max,
        "ecg_man": ecg_min,
        "ecgi_rng": ecgi_range,
        "ecgi_rrng": ecgi_rrange,
        "ecgi_max": ecgi_max,
        "ecgi_man": ecgi_min,
        "ecl_rng": ecl_range,
        "ecl_rrng": ecl_rrange,
        "ecl_max": ecl_max,
        "ecl_man": ecl_min,
        "u_mean": umean,
        "u_rng": urange,
        "u_max": umax,
        "u_man": umin,
    }
    corrs = cs.to_dict()["r"]
    all_stats = DataFrame({**{**stats, **corrs}}, index=[0])

    return means, all_stats


def get_step_ps(n_classes: int, rng: Generator) -> ndarray:
    max_width = ceil(n_classes / 5)
    n_steps = rng.integers(2, max(3, ceil(n_classes / 5) + 1))  # n_steps >= 2
    step_heights = np.random.uniform(0, 1, n_classes).tolist()
    steps = []
    step_widths: List[int] = []
    for i in range(n_steps):
        wmax = min(max_width, n_classes - np.sum(step_widths))
        width = int(rng.integers(2, wmax)) if wmax > 2 else 0
        steps.extend([step_heights[i] for _ in range(width)])
        step_widths.append(width)

    n_remain = n_classes - len(steps)
    p_remain = rng.uniform(0, 1, n_remain).tolist()
    ps: ndarray = np.array([*p_remain, *steps])
    ps = -np.sort(-ps) / ps.sum()
    return ps


def get_p(
    dist: Literal[
        "flat",
        "unif",
        "unif-r",
        "balanced",
        "balanced-r",
        "bimodal",
        "bimodal-r",
        "multimodal",
        "multimodal-r",
        "step",
        "step-r",
        "exp",
        "exp-r",
    ],
    n_classes: int,
    rng: Generator,
    n_modes: Optional[int] = None,
) -> Tuple[Optional[ndarray], Optional[int]]:
    if "flat" in dist:
        return None, n_modes
    elif "unif" in dist:
        p = rng.uniform(0, 1, size=n_classes)
        p /= p.sum()
        p = -np.sort(-p)
        return p, n_modes
    elif "bimodal" in dist:
        extreme = n_classes / 2
        p = np.ones([n_classes])
        p[0] = p[-1] = extreme
        p /= p.sum()
        p = -np.sort(-p)
    elif "balanced" in dist:
        p = rng.uniform(0.9, 1.1, size=n_classes)
        p /= p.sum()
        p = -np.sort(-p)
    elif "multi" in dist:
        n_modes = rng.integers(0, min(n_classes, 10)) if n_modes is None else n_modes
        extreme = n_classes / n_modes
        p = np.ones([n_classes])
        p[:n_modes] = extreme
        p /= p.sum()
        p = -np.sort(-p)
    elif "exp" in dist:
        # we want 1 / pmin**20 = 100, so that the majority class is 100 times more likely
        # than the minority. That is
        # pmin**20 = (1/100) = 10**-2
        # 20 * log10(pmin) = log10( 10**-2) = -2
        # log10(pmin) = -0.1
        # pmin = 10**-0.1 = 0.7943282347
        pmin = 0.7943282347
        scale = rng.uniform(0.1, 20)  # 0.1 is almost like a uniform, visually
        p = np.linspace(pmin, 1, n_classes) ** scale
        p /= p.sum()
        p = -np.sort(-p)
    elif "step" in dist:
        p = get_step_ps(n_classes=n_classes, rng=rng)
    else:
        raise ValueError()
    if "-r" in dist:
        p = np.sort(p)
    return p, n_modes


def rejection_sample(
    classes: List[int], size: int, p: Optional[ndarray], rng: Generator
) -> ndarray:
    """Use rejection sampling to throw out samples missing some class labels"""
    samples = rng.choice(classes, size=size, p=p)
    counts = np.bincount(samples, minlength=classes[-1] + 1)
    while np.any(counts == 0):
        samples = rng.choice(classes, size=size, p=p)
        counts = np.bincount(samples, minlength=classes[-1] + 1)
    return samples


def compare_error_styles(args: Namespace) -> Tuple[DataFrame, DataFrame]:
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
    edist = args.edist
    ydist = args.ydist
    error_style = args.errors
    n_classes = args.n_classes
    rng = np.random.default_rng(seed=args.seed)

    CLASSES = list(range(n_classes))
    yp, n_modes = get_p(ydist, n_classes, rng)
    # we really do need to ensure all classes are present in `y` for sim to make sense
    y = rejection_sample(classes=CLASSES, size=N, p=yp, rng=rng)
    ys = [y.copy() for _ in range(5)]
    n_max_err = ceil(s * len(y))
    err_max_idx = rng.permutation(len(y))[:n_max_err]
    if "multi" in ydist and "multi" in edist:
        # force same number of modes
        p = get_p(edist, n_classes, rng, n_modes=n_modes)[0]
    else:
        p = get_p(edist, n_classes, rng)[0]

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
        # Here, `r` essentially determines the independence. We do NOT use
        # rejection sampling here, because it need not be the case that an
        # algorithm always predicts at least one label from each class.
        pred_base = rng.choice(CLASSES, size=n_max_err, p=p)
        for yy in ys:
            pred_rand = rng.choice(CLASSES, size=n_max_err, p=p)
            # Large r means make more predictions random,  so less similar, and
            # more indepedent. Essentially, err_idx below determines which
            # samples are given random predictions.
            # So as r -> 1, almost all samples get a prediction from pred_base,
            # i.e. all predictions become the same, and are completely dependent.
            # As r -> 0, all predictions are completely random, so independent.

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
            "edist": edist,
            "ydist": ydist,
            "errors": error_style,
            "r": r,
            "s": s,
            "sr": s * r,
        },
        index=[0],
    )
    df, corrs = get_stats(y=y, ys=ys, y_errs=y_errs, err_idx=idxs, n_classes=n_classes)
    return pd.concat([extra, df], axis=1), corrs


def get_df(
    n_iter: int = 25000,
    mode: Literal["append", "overwrite", "cached"] = "cached",
    no_parallel: bool = False,
) -> DataFrame:
    ss = np.random.SeedSequence()
    seeds = ss.spawn(n_iter)

    GRID = [
        Namespace(**{**{"seed": seed}, **args})
        for args, seed in zip(
            ParameterSampler(
                {
                    "n_classes": list(range(2, 50)),
                    # "n_classes": list(range(2, 5)),
                    "errors": DEPENDENCES,
                    "edist": E_DIST_ORDER,
                    "ydist": Y_DIST_ORDER,
                    "r": uniform(),
                    "err_size": uniform(),
                    "reps": list(range(N_REP)),
                },
                n_iter=n_iter,
            ),
            seeds,
        )
    ]
    if mode == "append" and (not DF_OUT.exists()):
        mode = "overwrite"

    if DF_OUT.exists() and CORRS_OUT.exists() and mode == "cached":
        df_all = pd.read_parquet(DF_OUT)
        c_all = pd.read_parquet(CORRS_OUT)
    elif mode == "overwrite":
        if no_parallel:
            dfs, corrs = list(zip(*map(compare_error_styles, GRID)))
        else:
            dfs, corrs = list(zip(*process_map(compare_error_styles, GRID, chunksize=1)))
        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        corrs_all = pd.concat(corrs, axis=0, ignore_index=True)
        c_all = pd.concat(
            [
                df_all.loc[:, ["n_cls", "edist", "ydist", "errors", "r", "s", "sr"]],
                corrs_all,
            ],
            axis=1,
        )
        df_all.to_parquet(DF_OUT)
        c_all.to_parquet(CORRS_OUT)
    elif DF_OUT.exists() and CORRS_OUT.exists() and mode == "append":
        df_old = pd.read_parquet(DF_OUT)
        c_old = pd.read_parquet(CORRS_OUT)

        if no_parallel:
            dfs, corrs = list(zip(*map(compare_error_styles, GRID)))
        else:
            dfs, corrs = list(zip(*process_map(compare_error_styles, GRID, chunksize=1)))
        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        corrs_all = pd.concat(corrs, axis=0, ignore_index=True)
        c_all = pd.concat(
            [
                df_all.loc[:, ["n_cls", "edist", "ydist", "errors", "r", "s", "sr"]],
                corrs_all,
            ],
            axis=1,
        )

        df_all = pd.concat([df_old, df_all], axis=0, ignore_index=True)
        c_all = pd.concat([c_old, c_all], axis=0, ignore_index=True)
        df_all.to_parquet(DF_OUT)
        c_all.to_parquet(CORRS_OUT)
    else:
        raise ValueError("Missing pre-computed tables.")

    df = pd.concat(
        [
            df_all,
            c_all.drop(columns=["n_cls", "edist", "ydist", "errors", "r", "s", "sr"]),
        ],
        axis=1,
    )
    # correct the different meaning of "r" in the two cases
    inds = df["errors"] == "independent"
    df.loc[inds, "s"] = df.loc[inds, "sr"]
    df.loc[inds, "r"] = 1.0
    df["acc"] = df["a_mean"].apply(
        lambda acc: "<33%" if acc < 1 / 3 else (">66%" if acc > 2 / 3 else "33%-66%")
    )
    return df


def print_descriptions(df: DataFrame) -> None:
    dists = pd.get_dummies(df[["errors", "ydist", "edist"]])
    df = pd.concat(
        [
            df.loc[:, "n_cls"].to_frame(),
            dists,
            df.drop(columns=["errors", "ydist", "edist", "n_cls"]),
        ],
        axis=1,
    )
    # n_cols = len(df.columns)
    # fmt = ["0.3f" for _ in range(n_cols + 1)]
    # fmt[0] = "0.0f"
    # print(df.to_markdown(tablefmt="simple", floatfmt=fmt, index=False))
    pd.options.display.max_rows = 1000
    desc = df.groupby(["errors", "ydist", "edist"]).describe()
    print(desc.round(3).T)
    print("Min values")
    print(df.groupby(["errors", "yist", "edist"]).min().T.round(3))
    print("Max values")
    print(df.groupby(["errors", "yist", "edist"]).max().T.round(3))
    # print("EC / acc_mean correlation:", df.ec_q.corr(df.acc_mean))
    # print("EC / Kappa_y correlation:", df.ec.corr(df.k_y))
    # print("EC / Kappa_e correlation:", df.ec.corr(df.k_e))
    # print(df_all.groupby("raters").corr("pearson").round(3))
    # corrs = df.corr("pearson")
    print("=" * 80)
    print("Correlations taking into account distributions")
    print("=" * 80)
    cg = df.groupby(["errors", "ydist", "edist"]).corr("pearson")
    print(cg.round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))
    cols = list(filter(lambda c: "ec" in c, cg.columns))  # type: ignore
    print("EC correlations taking into account distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3).T)
    cols = list(filter(lambda c: "K" in c, cg.columns))  # type: ignore
    print("Kappa correlations taking into account distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3).T)

    print("=" * 80)
    print("Correlations ignoring distributions")
    print("=" * 80)
    cg = df.drop(columns=["ydist", "edist"]).groupby(["errors"]).corr("pearson")
    print(cg.round(3))
    corrs = df.corr("pearson")
    print(corrs.round(3))
    cols = list(filter(lambda c: "ec" in c, cg.columns))  # type: ignore
    print("EC correlations ignoring distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3).T)

    cols = list(filter(lambda c: "K" in c, cg.columns))  # type: ignore
    print("Kappa correlations ignoring distributions")
    for col in cols:
        print("\n", col.upper())
        print(cg[col].unstack().round(3).T)


def scatter_grid(
    df: DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    col: Optional[str] = "errors",
    col_order: Optional[List[str]] = None,
    row: Optional[str] = None,
    row_order: Optional[List[str]] = None,
    hue: Optional[str] = "r",
    size: Optional[str] = "n_cls",
    dependence: Optional[Literal["both", "dependent", "independent"]] = None,
    outdirname: Optional[str] = None,
    subplots_adjust: Optional[Dict[str, float]] = None,
    show: bool = False,
    **kwargs,
) -> None:
    sbn.set_style("darkgrid")
    df = df.rename(columns=RENAMES)
    if (dependence is not None) and ((col == "errors") or (row == "errors")):
        raise ValueError("Cannot restrict dependence and also show on grid")
    if dependence is None:
        dependence = "both"

    if dependence == "dependent":
        df = df.loc[df["errors"] == "dependent"]
    elif dependence == "independent":
        df = df.loc[df["errors"] == "independent"]

    x = RENAMES[x] if x in RENAMES else x
    y = RENAMES[y] if y in RENAMES else y
    hue = RENAMES[hue] if hue in RENAMES else hue
    if row == col:
        raise ValueError(
            "Cannot have `row` == `col`, will cause very inscrutable errors."
        )
    if row is None and col is not None:
        corrs = df.groupby(col)[x].corr(df[y])  # type: ignore
    elif row is not None and col is None:
        corrs = df.groupby(row)[x].corr(df[y])  # type: ignore
    elif row is not None and col is not None:
        corrs = df.groupby([row, col])[x].corr(df[y])  # type: ignore

    grid = sbn.relplot(
        data=df,
        x=x,
        y=y,
        row=RENAMES[row] if row in RENAMES else row,
        row_order=row_order,
        col=RENAMES[col] if col in RENAMES else col,
        col_order=col_order,
        hue=hue,
        # palette="rocket_r",
        # palette="crest",
        palette="flare",
        size=RENAMES[size] if size in RENAMES else size,
        **kwargs,
    )
    fig: Figure = plt.gcf()
    if hue is not None:
        suptitle = f"{y} vs {x} (by {hue})"
    else:
        suptitle = f"{y} vs {x}"
    if row is not None and col is not None:
        suptitle = f"{suptitle} [{row} x {col}]"
    elif row is not None and col is None:
        suptitle = f"{suptitle} [by {row}]"
    elif row is None and col is not None:
        suptitle = f"{suptitle} [by {col}]"
    if dependence != "both":
        suptitle += f" - {dependence} only"

    fig.suptitle(suptitle if title is None else title)
    ax: Axes
    for ax in grid.axes.flat:
        edist = ax.get_title().split(" ")[-1]
        if row is not None and col is not None:
            ydist = ax.get_title().split(" |")[0].split(" ")[-1]
            r = f"(Pearson's r={corrs[ydist][edist].round(3)})"  # type: ignore
        else:
            r = f"(Pearson's r={corrs[edist].round(3)})"  # type: ignore
        ax.set_title(f"{ax.get_title()}\n{r}")
    fig.tight_layout()
    if subplots_adjust is None:
        fig.subplots_adjust(right=0.85)
    else:
        fig.subplots_adjust(**subplots_adjust)

    if show:
        return plt.show()
    if title is None:
        title = suptitle
    outname = title.replace(" ", "_")
    outname = re.sub(r"[^a-zA-Z\d_]", "", outname)
    outdir = PLOTS if outdirname is None else ensure_dir(PLOTS / outdirname)
    outfile = outdir / f"{outname}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved plot to {outfile}")


def scatter_grid_p(args: Dict[str, Any]) -> None:
    return scatter_grid(**args)


def plot_metric_distributions(df: DataFrame) -> None:
    metrics = [RENAMES[m] if m in RENAMES else m for m in METRICS]
    dfr = df.rename(columns=RENAMES)
    df_metrics = pd.melt(
        dfr,
        id_vars=["errors", "edist", "ydist"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Metric value",
    )
    # This is very good, makes it easy to compare behaviour in dependent vs. independent
    grid = sbn.catplot(
        df_metrics,
        col="edist",
        col_order=E_DIST_ORDER,
        row="ydist",
        row_order=Y_DIST_ORDER,
        hue="errors",
        hue_order=["independent", "dependent"],
        x="Metric value",
        y="Metric",
        order=metrics,
        kind="violin",
        split=True,
        height=4,
        aspect=0.75,
        linewidth=0.5,
    )
    grid.set_titles("{row_name}-y vs {col_name}-errs")
    outdir = ensure_dir(PLOTS / "metric_distributions")
    outfile = outdir / "metric_distributions_by_y_err_dists_dependence.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved plot to {outfile}")

    # This is also very good, summarizes above
    grid = sbn.catplot(
        df_metrics,
        hue="errors",
        hue_order=["independent", "dependent"],
        y="Metric value",
        x="Metric",
        order=metrics,
        kind="violin",
        split=True,
        height=4,
        aspect=2,
        linewidth=0.5,
    )
    grid.set_titles("{row_name}-y vs {col_name}-errs")
    outdir = ensure_dir(PLOTS / "metric_distributions")
    outfile = outdir / "metric_distributions_by_dependence.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved plot to {outfile}")


################################################################################
# TODO: Plot only dependent case by dependency `r` for each metric, either dist
# or corellations to gauge true sensitivity to dependence
################################################################################


def print_core_tables(df: DataFrame) -> None:
    metrics = [RENAMES[m] if m in RENAMES else m for m in METRICS]
    dfr = df.rename(columns=RENAMES)
    desc = (
        dfr.groupby(["errors"])[metrics]
        .describe(percentiles=[0.025, 0.25, 0.75, 0.975])
        .T.unstack()
        .round(3)
    )

    for _metric in METRICS:
        metric = RENAMES[_metric] if _metric in RENAMES else _metric
        desc = dfr.groupby(["errors"])[metrics].describe().T.unstack()

        desc = dfr.groupby(["edist", "ydist", "errors"])[metric].describe(
            percentiles=[0.025, 0.975]
        )
        rng = desc["max"] - desc["min"]
        rrng = desc["97.5%"] - desc["2.5%"]
        ddf = DataFrame({"range": rng, "rrange": rrng}, index=desc.index)


def run_compare_styles(
    n_iter: int = 25000,
    mode: Literal["append", "overwrite", "cached"] = "cached",
    make_plots: bool = False,
    print_descs: bool = False,
    no_parallel: bool = False,
) -> None:
    df = get_df(n_iter=n_iter, mode=mode, no_parallel=no_parallel)
    if (not make_plots) and (not print_descs):
        return
    # print_descriptions(df)
    # scatter_grid(df=df, x="a_mean", y="K", title="Cohen's Kappa vs. Mean Accuracy")
    # scatter_grid(
    #     df=df, x="a_mean", y="alpha", title="Krippendorf's Alpha vs. Mean Accuracy"
    # )
    # scatter_grid(df=df, x="a_mean", y="ec_g", title="EC (global) vs. Mean Accuracy"

    # find those least related to accuracy
    # df.groupby("errors").corr()["a_mean"].abs().sort_values()

    # least related to dependence are e_ebv, e_corr or ec_r, e_v, and finally K

    args = list(
        ParameterGrid(
            dict(
                df=[df],
                x=["a_mean"],
                y=METRICS,
                col=["edist"],
                col_order=[E_DIST_ORDER],
                row=["ydist"],
                row_order=[Y_DIST_ORDER],
                markers=["errors"],
                hue=["mean_cls"],
                dependence=["dependent", "independent"],  # type: ignore
                outdirname=["by_dist"],
                subplots_adjust=[dict(right=0.92, top=0.92)],
                show=[False],
            )
        )
    )
    # process_map(scatter_grid_p, args)

    # plot relation to mean class probability
    args.extend(
        list(
            ParameterGrid(
                dict(
                    df=[df],
                    x=["mean_cls"],
                    y=METRICS,
                    col=["errors"],
                    col_order=[["independent", "dependent"]],
                    # col="acc",
                    # col_order=["<33%", "33%-66%", ">66%"],
                    hue=["r"],
                    size=["a_mean"],
                    outdirname=["by_dependency"],
                    show=[False],
                )
            )
        )
    )
    # process_map(scatter_grid_p, args)

    dfsmall = df[df["n_cls"] < 10]
    args.extend(
        list(
            ParameterGrid(
                dict(
                    df=[dfsmall],
                    x=["a_mean"],
                    y=METRICS,
                    col=["edist"],
                    col_order=[E_DIST_ORDER],
                    row=["ydist"],
                    row_order=[Y_DIST_ORDER],
                    markers=["errors"],
                    hue=["mean_cls"],
                    dependence=["dependent", "independent"],  # type: ignore
                    outdirname=["few_classes"],
                    subplots_adjust=[dict(right=0.92, top=0.92)],
                    show=[False],
                )
            )
        )
    )
    if make_plots:
        process_map(scatter_grid_p, args, desc="Creating scatterplots")
        plot_metric_distributions(df)

    if not print_descs:
        return

    metrics = [RENAMES[m] if m in RENAMES else m for m in METRICS]
    dfr = df.rename(columns=RENAMES)
    deps = pd.get_dummies(dfr["errors"]).loc[:, "independent"]
    dfr.errors = deps
    dfr.rename(columns={"errors": "is_independent"}, inplace=True)
    print(
        dfr.groupby(["is_independent", "ydist", "edist"])
        .corrwith(dfr["Mean Accuracy"])
        .loc[:, metrics]
    )
    df_metrics = pd.melt(
        dfr,
        id_vars=["is_independent", "edist", "ydist"],
        value_vars=metrics,
        var_name="Metric value",
    )
    dfd = pd.melt(
        dfr[dfr["is_independent"] == 0],
        id_vars=["edist", "ydist", "Error independence", "Error set max size"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Metric value",
    )
    cs = (
        dfr.groupby(["is_independent", "ydist", "edist"])
        .corrwith(dfr["Mean Accuracy"])
        .loc[:, metrics]
        .reset_index()
        .drop(columns=["ydist", "edist"])
        .groupby("is_independent")
        .describe(percentiles=[0.05, 0.25])
        .T
    )
    c_descs = cs.T.loc[:, (slice(None), ["mean", "min", "5%", "25%", "max"])]
    print("Metric correlations with (useless) mean accuracy:")
    print(c_descs.T.unstack())

    cs = (
        dfr.groupby(["is_independent", "ydist", "edist"])
        .corrwith(dfr["a_rng"])  # a_rrng no different really
        .loc[:, metrics]
        .reset_index()
        .drop(columns=["ydist", "edist"])
        .groupby("is_independent")
        .describe(percentiles=[0.05, 0.25])
        .T
    )
    c_descs = cs.T.loc[:, (slice(None), ["mean", "min", "5%", "25%", "max"])]
    print("Metric correlations with accuracy range:")
    print(c_descs.T.unstack())

    cs = (
        dfr.loc[dfr["is_independent"].groupby(["ydist", "edist"]) == 0]
        .corrwith(dfr["Error independence"])
        .loc[:, metrics]
        .reset_index()
        .drop(columns=["ydist", "edist"])
        .describe(percentiles=[0.05, 0.25])
        .T
    )
    print("Metric correlations with level of dependence:")
    print(cs.unstack())

    return

    # look at small number of classes only

    scatter_grid(
        df=df,
        x="a_mean",
        y="ec_l",
        hue="s",
        title="EC (local) vs. Mean Accuracy (by size)",
    )
    # scatter_grid(df=df, x="a_mean", y="e_v", title="Cramer's V vs. Mean Accuracy")
    # scatter_grid(df=df, x="ec_g", y="ec_l", title="EC (global) vs. EC (local)")


if __name__ == "__main__":
    # run_compare_raters()
    # run_compare_styles(n_iter=25000, mode="append")
    # run_compare_styles(n_iter=100_000, mode="cached")
    # MODE = "overwrite"
    MODE = "cached"
    run_compare_styles(
        n_iter=200_000,
        mode=MODE,
        make_plots=False,
        print_descs=True,
        no_parallel=False,
    )
