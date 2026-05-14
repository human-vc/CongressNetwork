import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, SEED

import pyfixest as pf

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, packages
from rpy2.robjects.conversion import localconverter

if not hasattr(np, "alltrue"):
    np.alltrue = np.all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HD = packages.importr("HonestDiD")
BASE = packages.importr("base")
STATS = packages.importr("stats")


def parse_event_time(coef_name):
    match = re.search(r"event_time::(-?\d+)", coef_name)
    if match:
        return int(match.group(1))
    return None


def extract_event_study(fit, term_filter):
    tidy = fit.tidy()
    rows = []
    for name, row in tidy.iterrows():
        if term_filter not in name:
            continue
        et = parse_event_time(name)
        if et is None:
            continue
        rows.append({"event_time": et, "coef_name": name, "estimate": float(row["Estimate"])})
    rows.sort(key=lambda r: r["event_time"])
    if not rows:
        raise ValueError(f"No coefficients matched term_filter='{term_filter}'")

    col_names = [str(c) for c in fit.coef().index]
    vcov_arr = np.asarray(fit._vcov, dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(col_names)}
    rows = [{**r, "coef_name": str(r["coef_name"])} for r in rows]

    indices = [name_to_idx[r["coef_name"]] for r in rows]
    betahat = np.array([r["estimate"] for r in rows], dtype=np.float64)
    sigma = vcov_arr[np.ix_(indices, indices)]

    event_times = [r["event_time"] for r in rows]
    num_pre = sum(1 for t in event_times if t < 0)
    num_post = sum(1 for t in event_times if t >= 0)
    return {
        "betahat": betahat,
        "sigma": sigma,
        "event_times": event_times,
        "num_pre": num_pre,
        "num_post": num_post,
        "coef_names": [r["coef_name"] for r in rows],
    }


def to_r_matrix(arr):
    arr = np.asarray(arr, dtype=np.float64)
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_obj = ro.conversion.py2rpy(arr)
    if arr.ndim == 1:
        return ro.r["matrix"](r_obj, nrow=len(arr), ncol=1)
    return ro.r["matrix"](r_obj, nrow=arr.shape[0], ncol=arr.shape[1])


def to_r_vector(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.py2rpy(arr)


def r_df_to_pandas(r_df):
    with localconverter(ro.default_converter):
        col_names = list(r_df.colnames)
        n = r_df.nrow
        data = {}
        for j, col in enumerate(col_names):
            r_col = r_df.rx2(col)
            try:
                data[col] = list(r_col)
            except Exception:
                data[col] = [r_col[i] for i in range(n)]
    return pd.DataFrame(data)


def original_cs(extracted, l_vec_index=0):
    betahat = to_r_vector(extracted["betahat"])
    sigma = to_r_matrix(extracted["sigma"])
    l_vec = HD.basisVector(int(l_vec_index + 1), extracted["num_post"])
    out = HD.constructOriginalCS(
        betahat=betahat,
        sigma=sigma,
        numPrePeriods=int(extracted["num_pre"]),
        numPostPeriods=int(extracted["num_post"]),
        l_vec=l_vec,
    )
    return r_df_to_pandas(out).iloc[0].to_dict()


def rm_sensitivity(extracted, mbar_vec, l_vec_index=0):
    betahat = to_r_vector(extracted["betahat"])
    sigma = to_r_matrix(extracted["sigma"])
    l_vec = HD.basisVector(int(l_vec_index + 1), extracted["num_post"])
    mbar_r = to_r_vector(mbar_vec)
    out = HD.createSensitivityResults_relativeMagnitudes(
        betahat=betahat,
        sigma=sigma,
        numPrePeriods=int(extracted["num_pre"]),
        numPostPeriods=int(extracted["num_post"]),
        Mbarvec=mbar_r,
        l_vec=l_vec,
    )
    return r_df_to_pandas(out)


def sd_sensitivity(extracted, m_vec, l_vec_index=0):
    betahat = to_r_vector(extracted["betahat"])
    sigma = to_r_matrix(extracted["sigma"])
    l_vec = HD.basisVector(int(l_vec_index + 1), extracted["num_post"])
    m_r = to_r_vector(m_vec)
    out = HD.createSensitivityResults(
        betahat=betahat,
        sigma=sigma,
        numPrePeriods=int(extracted["num_pre"]),
        numPostPeriods=int(extracted["num_post"]),
        Mvec=m_r,
        l_vec=l_vec,
    )
    return r_df_to_pandas(out)


def average_post_l_vec(num_post):
    arr = np.ones(num_post) / num_post
    return arr


def rm_sensitivity_average(extracted, mbar_vec):
    betahat = to_r_vector(extracted["betahat"])
    sigma = to_r_matrix(extracted["sigma"])
    l_vec = to_r_matrix(average_post_l_vec(extracted["num_post"]))
    mbar_r = to_r_vector(mbar_vec)
    out = HD.createSensitivityResults_relativeMagnitudes(
        betahat=betahat,
        sigma=sigma,
        numPrePeriods=int(extracted["num_pre"]),
        numPostPeriods=int(extracted["num_post"]),
        Mbarvec=mbar_r,
        l_vec=l_vec,
    )
    return r_df_to_pandas(out)


def fit_cohort_ddd_event_study():
    panel_path = RESULTS_DIR / "cohort_ddd_panel.csv"
    df = pd.read_csv(panel_path)
    df = df[df["freshman"] == 1].copy()
    df["event_time"] = df["event_time"].astype(int)
    df["competitive"] = df["competitive"].astype(float)
    fit = pf.feols(
        "cross_party_agreement ~ i(event_time, competitive, ref=-1) | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )
    return fit


def fit_staggered_wave_event_study(wave=112):
    stacked = pd.read_csv(RESULTS_DIR / "staggered_stacked_panel.csv")
    sub = stacked[stacked["stack_id"] == wave].copy()
    sub["event_time"] = sub["event_time"].astype(int)
    sub["treated"] = sub["treated"].astype(float)
    fit = pf.feols(
        "cross_party_agreement ~ i(event_time, treated, ref=-1) | district_id + congress",
        data=sub,
        vcov={"CRV1": "district_id"},
    )
    return fit


def plot_sensitivity(rm_df, sd_df, original_dict, out_path, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if "Mbar" in rm_df.columns:
        ax.plot(rm_df["Mbar"], rm_df["lb"], "o-", color="#1f77b4", label="lower")
        ax.plot(rm_df["Mbar"], rm_df["ub"], "o-", color="#1f77b4", label="upper")
        ax.fill_between(rm_df["Mbar"], rm_df["lb"], rm_df["ub"], alpha=0.2, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(original_dict["lb"], color="red", linestyle="--", alpha=0.6, label="original lb")
    ax.axhline(original_dict["ub"], color="red", linestyle="--", alpha=0.6, label="original ub")
    ax.set_xlabel(r"$\bar{M}$ (relative magnitudes)")
    ax.set_ylabel("Robust CI")
    ax.set_title("Relative magnitudes bounds")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    if "M" in sd_df.columns:
        ax.plot(sd_df["M"], sd_df["lb"], "o-", color="#2ca02c", label="lower")
        ax.plot(sd_df["M"], sd_df["ub"], "o-", color="#2ca02c", label="upper")
        ax.fill_between(sd_df["M"], sd_df["lb"], sd_df["ub"], alpha=0.2, color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(original_dict["lb"], color="red", linestyle="--", alpha=0.6, label="original lb")
    ax.axhline(original_dict["ub"], color="red", linestyle="--", alpha=0.6, label="original ub")
    ax.set_xlabel(r"$M$ (smoothness)")
    ax.set_ylabel("Robust CI")
    ax.set_title("Smoothness bounds")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def analyze_fit(fit, term_filter, label, mbar_vec, m_vec, figure_path):
    extracted = extract_event_study(fit, term_filter)
    orig = original_cs(extracted, l_vec_index=0)
    rm_zero = rm_sensitivity(extracted, mbar_vec, l_vec_index=0)
    sd_zero = sd_sensitivity(extracted, m_vec, l_vec_index=0)
    rm_avg = rm_sensitivity_average(extracted, mbar_vec) if extracted["num_post"] > 1 else None

    plot_sensitivity(rm_zero, sd_zero, orig, figure_path, label)

    result = {
        "label": label,
        "num_pre_periods": extracted["num_pre"],
        "num_post_periods": extracted["num_post"],
        "event_times": extracted["event_times"],
        "betahat": extracted["betahat"].tolist(),
        "original_cs_event0": orig,
        "rm_bounds_event0": rm_zero.to_dict(orient="records"),
        "sd_bounds_event0": sd_zero.to_dict(orient="records"),
    }
    if rm_avg is not None:
        result["rm_bounds_average"] = rm_avg.to_dict(orient="records")

    breakdown = None
    for row in result["rm_bounds_event0"]:
        if row.get("lb", 0) <= 0 <= row.get("ub", 0):
            breakdown = row.get("Mbar")
            break
    result["breakdown_mbar"] = breakdown
    return result


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cohort_fit = fit_cohort_ddd_event_study()
    cohort_result = analyze_fit(
        cohort_fit,
        term_filter=":competitive",
        label="cohort_ddd_event_study",
        mbar_vec=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        m_vec=[0.0, 0.005, 0.01, 0.02, 0.05],
        figure_path=FIGURES_DIR / "honestdid_cohort_ddd.pdf",
    )

    wave112_result = {"note": "Wave 112 has only 1 pre-period after cycle constraint; HonestDiD requires >=2 pre-periods for meaningful bounds; see cohort DDD result for sensitivity defense"}
    try:
        wave112_fit = fit_staggered_wave_event_study(wave=112)
        wave112_extracted = extract_event_study(wave112_fit, ":treated")
        wave112_result["num_pre_periods"] = wave112_extracted["num_pre"]
        wave112_result["num_post_periods"] = wave112_extracted["num_post"]
        wave112_result["betahat"] = wave112_extracted["betahat"].tolist()
        if wave112_extracted["num_pre"] >= 2:
            wave112_result.update(analyze_fit(
                wave112_fit,
                term_filter=":treated",
                label="wave_112_tea_party",
                mbar_vec=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
                m_vec=[0.0, 0.005, 0.01, 0.02, 0.05],
                figure_path=FIGURES_DIR / "honestdid_wave_112.pdf",
            ))
    except Exception as e:
        wave112_result["error"] = str(e)

    output = {
        "cohort_ddd": cohort_result,
        "wave_112_tea_party": wave112_result,
    }

    out_path = RESULTS_DIR / "honest_did_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Cohort DDD: pre={cohort_result['num_pre_periods']}, post={cohort_result['num_post_periods']}")
    print(f"  Original CS at event 0: {cohort_result['original_cs_event0']}")
    print(f"  Breakdown Mbar: {cohort_result['breakdown_mbar']}")
    print()
    print(f"Wave 112: pre={wave112_result.get('num_pre_periods')}, post={wave112_result.get('num_post_periods')}")
    if "original_cs_event0" in wave112_result:
        print(f"  Original CS at event 0: {wave112_result['original_cs_event0']}")
        print(f"  Breakdown Mbar: {wave112_result['breakdown_mbar']}")
    else:
        print(f"  Note: {wave112_result.get('note', wave112_result.get('error'))}")
    print()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
