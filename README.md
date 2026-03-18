# ============================================================
# equine_interaction_reproducible_analysis.py
# Fully reproducible analysis script for the equine interaction study
# Python 3.11
# Packages: pandas, numpy, scipy, statsmodels
#
# To reproduce results:
# 1. Place the dataset in the same directory as this script
#    using the filename: equine_interaction_dataset.csv
# 2. Run: python equine_interaction_reproducible_analysis.py
# ============================================================

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

FILE = Path("equine_interaction_dataset.csv")
if not FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {FILE.resolve()}\n"
        "Place 'equine_interaction_dataset.csv' in the same directory as this script."
    )

df = pd.read_csv(FILE)

# Standardize condition coding
df["condition"] = df["condition"].astype(str).str.strip()
df["condition_bin"] = df["condition"].map({"Petting": 0, "Scratching": 1})

if df["condition_bin"].isna().any():
    bad_vals = sorted(df.loc[df["condition_bin"].isna(), "condition"].dropna().unique())
    raise ValueError(
        "Unrecognized values detected in 'condition': "
        f"{bad_vals}. Expected only 'Petting' and 'Scratching'."
    )

# Ensure week is numeric and compatible with statsmodels formulas
df["week"] = pd.to_numeric(df["week"], errors="coerce")

# ------------------------------------------------------------
# 2. BASIC CLEANING
# ------------------------------------------------------------

emotion_vars = ["nervous", "sad", "excited", "lonely", "calm"]

# Complete cases for emotional analyses
emotion_complete = df.dropna(
    subset=["participant_id", "week"]
    + [f"{v}_pre" for v in emotion_vars]
    + [f"{v}_post" for v in emotion_vars]
).copy()

# Heart rate cleaning
for col in ["hr_pre", "hr_post"]:
    if col in df.columns:
        df.loc[(df[col] < 40) | (df[col] > 180), col] = np.nan

# Complete cases for behavioral analyses
behavior_complete = df.dropna(
    subset=["participant_id", "condition_bin", "week", "any_lip", "any_walk"]
).copy()

# Convert behavior indicators to numeric
behavior_complete["any_lip"] = pd.to_numeric(
    behavior_complete["any_lip"], errors="coerce"
)
behavior_complete["any_walk"] = pd.to_numeric(
    behavior_complete["any_walk"], errors="coerce"
)

# Final cleanup after coercion
behavior_complete = behavior_complete.dropna(
    subset=["participant_id", "condition_bin", "week", "any_lip", "any_walk"]
).copy()

# Use integer coding where appropriate
behavior_complete["condition_bin"] = behavior_complete["condition_bin"].astype(int)
behavior_complete["any_lip"] = behavior_complete["any_lip"].astype(int)
behavior_complete["any_walk"] = behavior_complete["any_walk"].astype(int)

# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def descriptives_pre_post(data: pd.DataFrame, var: str) -> dict:
    """Return descriptive statistics for paired pre/post measures."""
    pre = data[f"{var}_pre"].dropna()
    post = data[f"{var}_post"].dropna()
    return {
        "pre_mean": pre.mean(),
        "pre_sd": pre.std(ddof=1),
        "post_mean": post.mean(),
        "post_sd": post.std(ddof=1),
        "n": min(len(pre), len(post)),
    }


def paired_t_test(data: pd.DataFrame, var: str) -> dict:
    """Paired t-test for pre/post change."""
    sub = data[[f"{var}_pre", f"{var}_post"]].dropna()
    t_stat, p_val = stats.ttest_rel(sub[f"{var}_pre"], sub[f"{var}_post"])
    return {
        "t": t_stat,
        "df": len(sub) - 1,
        "p": p_val,
    }


def make_long(data: pd.DataFrame, var: str) -> pd.DataFrame:
    """Reshape one emotion variable to long format for mixed modeling."""
    sub = data[["participant_id", "week", f"{var}_pre", f"{var}_post"]].copy()
    long_df = pd.melt(
        sub,
        id_vars=["participant_id", "week"],
        value_vars=[f"{var}_pre", f"{var}_post"],
        var_name="time",
        value_name="score",
    )
    long_df["time"] = long_df["time"].map({f"{var}_pre": 0, f"{var}_post": 1})
    return long_df.dropna()


def fit_lmm_pre_post(data: pd.DataFrame, var: str):
    """
    Linear mixed model:
    score ~ time
    random intercept for participant
    additional variance component for week
    """
    long_df = make_long(data, var)

    model = smf.mixedlm(
        "score ~ time",
        data=long_df,
        groups=long_df["participant_id"],
        vc_formula={"week": "0 + C(week)"},
    )
    result = model.fit(reml=False, method="lbfgs")
    return result


def fit_clustered_logit(
    data: pd.DataFrame, outcome: str, direction: str = "positive"
) -> dict:
    """
    Logistic regression with cluster-robust SE by week.

    Parameters
    ----------
    outcome : str
        'any_lip' or 'any_walk'
    direction : str
        'positive' => scratching increases outcome
        'negative' => scratching decreases outcome

    Returns
    -------
    dict
        Model object, coefficients, OR, CIs, descriptive percentages, and p-values.
    """
    sub = data.dropna(subset=[outcome, "condition_bin", "week"]).copy()

    model = smf.glm(
        formula=f"{outcome} ~ condition_bin",
        data=sub,
        family=sm.families.Binomial(),
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sub["week"]})

    beta = result.params["condition_bin"]
    se = result.bse["condition_bin"]
    z_stat = beta / se
    p_two = result.pvalues["condition_bin"]

    if direction == "positive":
        p_one = stats.norm.sf(z_stat)
    elif direction == "negative":
        p_one = stats.norm.cdf(z_stat)
    else:
        raise ValueError("direction must be 'positive' or 'negative'")

    odds_ratio = np.exp(beta)
    ci_low = np.exp(beta - 1.96 * se)
    ci_high = np.exp(beta + 1.96 * se)

    pct = (
        sub.groupby("condition")[outcome]
        .mean()
        .mul(100)
        .round(1)
        .to_dict()
    )

    return {
        "model": result,
        "beta": beta,
        "se": se,
        "z": z_stat,
        "p_two": p_two,
        "p_one": p_one,
        "OR": odds_ratio,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "percentages": pct,
        "n": len(sub),
    }


def fit_baseline_adjusted_model(data: pd.DataFrame, outcome: str):
    """
    Baseline-adjusted linear model with cluster-robust SE by week:
    outcome_post ~ outcome_pre + any_lip + any_walk
    """
    sub = data.dropna(
        subset=[
            f"{outcome}_pre",
            f"{outcome}_post",
            "any_lip",
            "any_walk",
            "week",
        ]
    ).copy()

    model = smf.ols(
        formula=f"{outcome}_post ~ {outcome}_pre + any_lip + any_walk",
        data=sub,
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sub["week"]})
    return result


def summarize_coef(result, term: str) -> dict:
    """Return a concise summary for one model coefficient."""
    ci = result.conf_int()
    return {
        "beta": result.params.get(term, np.nan),
        "se": result.bse.get(term, np.nan),
        "t_or_z": result.tvalues.get(term, np.nan),
        "p": result.pvalues.get(term, np.nan),
        "ci_low": ci.loc[term, 0] if term in result.params.index else np.nan,
        "ci_high": ci.loc[term, 1] if term in result.params.index else np.nan,
    }


# ------------------------------------------------------------
# 4. QUESTION 1:
#    DID THE EQUINE INTERACTION ALTER EMOTIONAL STATE?
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION 1. PRE/POST CHANGES IN EMOTIONAL STATE")
print("=" * 70)

q1_results = {}

for var in emotion_vars:
    desc = descriptives_pre_post(emotion_complete, var)
    ttest = paired_t_test(emotion_complete, var)
    lmm = fit_lmm_pre_post(emotion_complete, var)

    q1_results[var] = {
        "descriptives": desc,
        "paired_t": ttest,
        "lmm_time_beta": lmm.params.get("time", np.nan),
        "lmm_time_se": lmm.bse.get("time", np.nan),
        "lmm_time_p": lmm.pvalues.get("time", np.nan),
        "lmm_object": lmm,
    }

    print(f"\nOutcome: {var}")
    print(f"  Pre:  M = {desc['pre_mean']:.2f}, SD = {desc['pre_sd']:.2f}")
    print(f"  Post: M = {desc['post_mean']:.2f}, SD = {desc['post_sd']:.2f}")
    print(f"  Paired t({ttest['df']}) = {ttest['t']:.2f}, p = {ttest['p']:.3g}")
    print(
        f"  LMM time effect (post vs pre): beta = "
        f"{q1_results[var]['lmm_time_beta']:.3f}, "
        f"SE = {q1_results[var]['lmm_time_se']:.3f}, "
        f"p = {q1_results[var]['lmm_time_p']:.3g}"
    )

# ------------------------------------------------------------
# 5. QUESTION 2:
#    DID TACTILE MANIPULATION INFLUENCE EQUINE BEHAVIOR?
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION 2. TACTILE MANIPULATION -> EQUINE BEHAVIOR")
print("=" * 70)

lip_res = fit_clustered_logit(
    behavior_complete,
    outcome="any_lip",
    direction="positive",
)

walk_res = fit_clustered_logit(
    behavior_complete,
    outcome="any_walk",
    direction="negative",
)

print("\nOutcome: any_lip")
print(f"  n = {lip_res['n']}")
print(f"  Percent observed by condition: {lip_res['percentages']}")
print(f"  beta = {lip_res['beta']:.3f}, SE = {lip_res['se']:.3f}, z = {lip_res['z']:.3f}")
print(
    f"  OR = {lip_res['OR']:.3f}, "
    f"95% CI [{lip_res['CI_low']:.3f}, {lip_res['CI_high']:.3f}]"
)
print(f"  two-tailed p = {lip_res['p_two']:.3g}")
print(f"  one-tailed p = {lip_res['p_one']:.3g}")

print("\nOutcome: any_walk")
print(f"  n = {walk_res['n']}")
print(f"  Percent observed by condition: {walk_res['percentages']}")
print(
    f"  beta = {walk_res['beta']:.3f}, SE = {walk_res['se']:.3f}, "
    f"z = {walk_res['z']:.3f}"
)
print(
    f"  OR = {walk_res['OR']:.3f}, "
    f"95% CI [{walk_res['CI_low']:.3f}, {walk_res['CI_high']:.3f}]"
)
print(f"  two-tailed p = {walk_res['p_two']:.3g}")
print(f"  one-tailed p = {walk_res['p_one']:.3g}")

# ------------------------------------------------------------
# 6. QUESTION 3:
#    DOES EQUINE ENGAGEMENT PREDICT HUMAN EMOTIONAL OUTCOMES?
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION 3. BASELINE-ADJUSTED MODELS (TWO-WAY HYPOTHESIS)")
print("=" * 70)

q3_results = {}

for var in emotion_vars:
    res = fit_baseline_adjusted_model(behavior_complete, var)
    q3_results[var] = {
        "model": res,
        "pre": summarize_coef(res, f"{var}_pre"),
        "any_lip": summarize_coef(res, "any_lip"),
        "any_walk": summarize_coef(res, "any_walk"),
        "n": int(res.nobs),
    }

    print(f"\nOutcome: {var}_post")
    print(f"  n = {int(res.nobs)}")
    print(
        f"  Baseline ({var}_pre): beta = {q3_results[var]['pre']['beta']:.3f}, "
        f"SE = {q3_results[var]['pre']['se']:.3f}, "
        f"p = {q3_results[var]['pre']['p']:.3g}"
    )
    print(
        f"  any_lip:  beta = {q3_results[var]['any_lip']['beta']:.3f}, "
        f"SE = {q3_results[var]['any_lip']['se']:.3f}, "
        f"p = {q3_results[var]['any_lip']['p']:.3g}"
    )
    print(
        f"  any_walk: beta = {q3_results[var]['any_walk']['beta']:.3f}, "
        f"SE = {q3_results[var]['any_walk']['se']:.3f}, "
        f"p = {q3_results[var]['any_walk']['p']:.3g}"
    )

# ------------------------------------------------------------
# 7. OPTIONAL: HEART RATE PRE/POST
# ------------------------------------------------------------

if {"hr_pre", "hr_post"}.issubset(df.columns):
    print("\n" + "=" * 70)
    print("OPTIONAL. HEART RATE PRE/POST")
    print("=" * 70)

    hr_df = df.dropna(subset=["participant_id", "week", "hr_pre", "hr_post"]).copy()
    hr_long = pd.melt(
        hr_df[["participant_id", "week", "hr_pre", "hr_post"]],
        id_vars=["participant_id", "week"],
        value_vars=["hr_pre", "hr_post"],
        var_name="time",
        value_name="hr",
    )
    hr_long["time"] = hr_long["time"].map({"hr_pre": 0, "hr_post": 1})

    hr_model = smf.mixedlm(
        "hr ~ time",
        data=hr_long,
        groups=hr_long["participant_id"],
        vc_formula={"week": "0 + C(week)"},
    )
    hr_result = hr_model.fit(reml=False, method="lbfgs")

    hr_t = stats.ttest_rel(hr_df["hr_pre"], hr_df["hr_post"])

    print(
        f"Pre HR:  M = {hr_df['hr_pre'].mean():.2f}, "
        f"SD = {hr_df['hr_pre'].std(ddof=1):.2f}"
    )
    print(
        f"Post HR: M = {hr_df['hr_post'].mean():.2f}, "
        f"SD = {hr_df['hr_post'].std(ddof=1):.2f}"
    )
    print(f"Paired t({len(hr_df) - 1}) = {hr_t.statistic:.2f}, p = {hr_t.pvalue:.3g}")
    print(
        f"LMM time effect: beta = {hr_result.params['time']:.3f}, "
        f"SE = {hr_result.bse['time']:.3f}, "
        f"p = {hr_result.pvalues['time']:.3g}"
    )

# ------------------------------------------------------------
# 8. SAVE KEY RESULTS TABLES
# ------------------------------------------------------------

# Q1 summary table
q1_rows = []
for var, res in q1_results.items():
    q1_rows.append(
        {
            "outcome": var,
            "n": res["descriptives"]["n"],
            "pre_mean": res["descriptives"]["pre_mean"],
            "pre_sd": res["descriptives"]["pre_sd"],
            "post_mean": res["descriptives"]["post_mean"],
            "post_sd": res["descriptives"]["post_sd"],
            "paired_t": res["paired_t"]["t"],
            "paired_df": res["paired_t"]["df"],
            "paired_p": res["paired_t"]["p"],
            "lmm_time_beta": res["lmm_time_beta"],
            "lmm_time_se": res["lmm_time_se"],
            "lmm_time_p": res["lmm_time_p"],
        }
    )

q1_table = pd.DataFrame(q1_rows)
q1_table.to_csv("q1_prepost_emotion_results.csv", index=False)

# Q2 summary table
q2_table = pd.DataFrame(
    [
        {
            "outcome": "any_lip",
            "n": lip_res["n"],
            "petting_pct": lip_res["percentages"].get("Petting", np.nan),
            "scratching_pct": lip_res["percentages"].get("Scratching", np.nan),
            "beta": lip_res["beta"],
            "se": lip_res["se"],
            "z": lip_res["z"],
            "OR": lip_res["OR"],
            "CI_low": lip_res["CI_low"],
            "CI_high": lip_res["CI_high"],
            "p_two": lip_res["p_two"],
            "p_one": lip_res["p_one"],
        },
        {
            "outcome": "any_walk",
            "n": walk_res["n"],
            "petting_pct": walk_res["percentages"].get("Petting", np.nan),
            "scratching_pct": walk_res["percentages"].get("Scratching", np.nan),
            "beta": walk_res["beta"],
            "se": walk_res["se"],
            "z": walk_res["z"],
            "OR": walk_res["OR"],
            "CI_low": walk_res["CI_low"],
            "CI_high": walk_res["CI_high"],
            "p_two": walk_res["p_two"],
            "p_one": walk_res["p_one"],
        },
    ]
)
q2_table.to_csv("q2_tactile_behavior_results.csv", index=False)

# Q3 summary table
q3_rows = []
for var, res in q3_results.items():
    q3_rows.append(
        {
            "outcome": var,
            "n": res["n"],
            "beta_pre": res["pre"]["beta"],
            "se_pre": res["pre"]["se"],
            "p_pre": res["pre"]["p"],
            "beta_any_lip": res["any_lip"]["beta"],
            "se_any_lip": res["any_lip"]["se"],
            "p_any_lip": res["any_lip"]["p"],
            "beta_any_walk": res["any_walk"]["beta"],
            "se_any_walk": res["any_walk"]["se"],
            "p_any_walk": res["any_walk"]["p"],
        }
    )

q3_table = pd.DataFrame(q3_rows)
q3_table.to_csv("q3_two_way_results.csv", index=False)

print("\nSaved:")
print("  q1_prepost_emotion_results.csv")
print("  q2_tactile_behavior_results.csv")
print("  q3_two_way_results.csv")
<img width="468" height="645" alt="image" src="https://github.com/user-attachments/assets/f3cd43d2-86e2-494e-8614-717abda60ca1" />
