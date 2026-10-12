"""
make_figures.py — README figures: severity tiers, career-ending risk factors, model accuracy and the pricing back-test.

    python -m runs.make_figures        # writes figures/severity_tiers.png, career_risk_factors.png, model_accuracy.png,
                                       # pricing_backtest.png

Coefficients come from the fits on training players (80%); accuracy is scored on held-out players (20%).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from model import career
from model import severity as sev

OUT = Path(__file__).parent / "figures"
GLM, BASE = "#2a78d6", "#a3a29c"                       # model blue, baseline gray
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
times = FuncFormatter(lambda x, _: f"{x:g}×")


def _title(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.45 / h, subtitle, fontsize=10.5, color=INK_2, va="top")


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}")


KIND = {"Strain / muscle": "strain", "Tear / rupture": "tear", "Knock / bruise": "knock", "Fracture": "fracture",
        "Ligament / joint": "ligament", "Tendon / inflammation": "tendon", "Surgery": "surgery",
        "Concussion": "concussion"}


def _short(cls):
    """Rating class -> plain label: 'Knee - Tear / rupture' -> 'Knee tear'."""
    s = cls.replace(" (unspecified)", "")
    if " - " not in s:
        return "Unspecified injury" if s == "Unknown" else s
    site, kind = s.split(" - ", 1)
    if kind == "other sites":
        return f"Other {KIND[site]}s".replace("ys", "ies") if site in KIND else "Other unspecified injuries"
    if kind == "other types":
        return f"Other {site.lower()} injuries"
    if site == "Unknown":
        return f"{KIND[kind].capitalize()} (site unknown)"
    return f"{site} {KIND[kind]}"


# --------------------------------------------------------------------------
# Severity tiers
# --------------------------------------------------------------------------
def severity_figure(model, train):
    coef = sev.coefficients(model)["coef"]
    d = sev.completed(train).assign(tier=lambda x: x["injury_class"].astype(str).map(model.tier_of))
    rows = []
    for t in model.tier_names:
        sub = d[d["tier"] == t]
        top = sub["injury_class"].astype(str).value_counts().index[:2]
        rows.append({"tier": t, "mult": np.exp(coef.get(f"severity_tier = {t}", 0.0)),
                     "median": sub["days_missed"].median(), "n": len(sub),
                     "examples": ", ".join(_short(c) for c in top)})
    tiers = pd.DataFrame(rows).iloc[::-1]
    y = np.arange(len(tiers))

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.86, bottom=0.08)
    ref = tiers["tier"] == model.ref_tier
    ax.barh(y, tiers["mult"] - 1, left=1, height=0.62, color=np.where(ref, INK_2, GLM))
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 8)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.xaxis.set_major_formatter(times)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y, [f"{t}  ·  {e}" for t, e in zip(tiers["tier"], tiers["examples"])])
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10)
    ax.set_xlabel(f"Expected days missed vs {model.ref_tier} (hamstring), same player (log scale)")
    for yi, r in zip(y, tiers.itertuples()):
        x = max(r.mult, 1) * 1.06
        ax.text(x, yi, f"{r.mult:.2f}×   {r.median:.0f} d", va="center", fontsize=9.5, color=INK)
    ax.text(-0.01, 1.01, "Tier  ·  most common injuries", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=INK_2)
    ax.text(0.99, 1.01, "label: multiplier   median days", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=INK_2)
    _title(fig, "Injury type sets the expected layoff: 16 severity tiers",
           "Gamma GLM multiplier on expected days missed (bars) and median actual days per tier.\n"
           f"Fit on training players ({len(d):,} completed injuries); "
           "player age, position and injury history held fixed.")
    _save(fig, "severity_tiers.png")
    return tiers


# --------------------------------------------------------------------------
# Career-ending odds ratios
# --------------------------------------------------------------------------
# (label, term, scale): odds ratio for scale x the coefficient; log1p terms use ln 2 = "doubling"
CAREER_TERMS = [
    ("Body part", [("Knee", "region_g = Knee", 1), ("Achilles", "region_g = Achilles", 1),
                   ("Body part not recorded", "region_g = Unknown", 1)]),
    ("Injury type", [("Surgery", "nature_g = Surgery", 1), ("Tear / rupture", "nature_g = Tear / rupture", 1),
                     ("Ligament / joint", "nature_g = Ligament / joint", 1)]),
    ("Position", [("Defender", "position_group = Defender", 1), ("Goalkeeper", "position_group = Goalkeeper", 1),
                  ("Attacker", "position_group = Attack", 1)]),
    ("Player", [("Age 31 vs 26", {"age_c": 5, "I(age_c ** 2)": 25}, 1),
                ("2× games played, last 12 months", "log(1 + games_12m)", np.log(2)),
                ("No games, last 12 months", "no_games_12m", 1),
                ("2× career games", "log(1 + career_games)", np.log(2)),
                ("2× prior injuries", "log(1 + prior_injuries)", np.log(2)),
                ("Injury year +1", "year_c", 1)]),
]
REFS = {"Body part": "vs hamstring", "Injury type": "vs unspecified", "Position": "vs midfielder", "Player": ""}


def career_figure(model, train):
    res = model.res
    names = dict(zip(sev.tidy_terms(res.params.index), res.params.index))
    rows, labels, groups = [], [], []
    for group, terms in CAREER_TERMS:
        for label, term, k in terms:
            w = pd.Series({names[t]: k * v for t, v in (term if isinstance(term, dict) else {term: 1}).items()})
            b, se = w @ res.params[w.index], np.sqrt(w @ res.cov_params().loc[w.index, w.index] @ w)
            rows.append(np.exp([b, b - 1.96 * se, b + 1.96 * se]))
            labels.append(label)
            groups.append(group)
    rows = np.array(rows)
    gap = np.cumsum([0] + [1.0 if g != groups[i - 1] else 0 for i, g in enumerate(groups) if i])
    y = -(np.arange(len(rows)) + gap * 0.9)

    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    fig.subplots_adjust(left=0.30, right=0.82, top=0.86, bottom=0.09)
    ax.axvline(1, color=INK, lw=1)
    ax.hlines(y, rows[:, 1], rows[:, 2], color=GLM, lw=2)
    ax.plot(rows[:, 0], y, "o", ms=8, color=GLM, mec="white", mew=2)
    for yi, (or_, lo, hi) in zip(y, rows):
        ax.text(1.02, yi, f"{or_:.2f}  ({lo:.2f}–{hi:.2f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9.5)
    ax.text(1.02, y.max() + 0.85, "Odds ratio (95% CI)", transform=ax.get_yaxis_transform(),
            va="center", fontsize=10, fontweight="bold", color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 30)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(times)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    for group in REFS:
        yg = y[[g == group for g in groups]].max() + 0.85
        ax.text(-0.02, yg, f"{group}  {REFS[group]}".strip(), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK_2)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.4)
    ax.set_xlabel("Odds ratio for a career-ending injury (log scale; 1 = no effect)")
    ax.text(1.04, y.max() + 0.85, "higher risk →", fontsize=9.5, color=INK_2, va="center")
    ax.text(0.96, y.max() + 0.85, "← lower risk", fontsize=9.5, color=INK_2, va="center", ha="right")
    _title(fig, "What makes an injury career-ending",
           "Logistic GLM odds ratios with 95% confidence intervals, all predictors adjusted for each other.\n"
           f"Fit on training players ({len(train):,} injuries, {int(train['career_ending'].sum()):,} career-ending).")
    _save(fig, "career_risk_factors.png")
    return pd.DataFrame(rows, index=labels, columns=["odds_ratio", "ci_low", "ci_high"])


# --------------------------------------------------------------------------
# Held-out accuracy
# --------------------------------------------------------------------------
def accuracy_figure(sev_models, sev_test, car_models, car_test):
    s = sev.compare(sev_models, sev_test)
    c = career.compare(car_models, car_test)
    mae = [s.loc["Gamma, no predictors", "mae"], s.loc["Gamma GLM, severity tiers", "mae"]]
    auc = c.loc["Logistic GLM, main effects", "auc"]
    top10 = c.loc["Logistic GLM, main effects", "top10_capture"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.set_figheight(5.2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.75, bottom=0.14, wspace=0.35)

    a1.bar([0, 1], mae, width=0.55, color=[BASE, GLM])
    for x, v in enumerate(mae):
        a1.text(x, v + 1, f"{v:.1f} days", ha="center", va="bottom", fontsize=11, fontweight="bold")
    a1.set_xticks([0, 1], ["No predictors\n(average layoff)", "Gamma GLM\n(severity tiers)"])
    a1.set_ylim(0, max(mae) * 1.22)
    a1.set_ylabel("Mean absolute error (days)")
    a1.set_title("Days missed: error per injury", pad=10)
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.set_axisbelow(True)

    y = car_test["career_ending"].to_numpy()
    p = car_models["Logistic GLM, main effects"].predict(car_test)
    order = np.argsort(-p)
    flagged = np.arange(1, len(y) + 1) / len(y)
    captured = np.cumsum(y[order]) / y.sum()
    a2.plot([0, 1], [0, 1], color=BASE, lw=2, ls="--")
    a2.plot(flagged, captured, color=GLM, lw=2)
    a2.plot([0.1], [top10], "o", ms=8, color=GLM, mec="white", mew=2)
    a2.annotate(f"Riskiest 10% of injuries\ncontain {top10:.0%} of career endings", (0.1, top10),
                xytext=(0.3, 0.38), fontsize=10, arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1})
    a2.text(0.62, 0.52, "No predictors (AUC 0.50)", color=INK_2, fontsize=9.5, rotation=0)
    a2.text(0.33, 0.74, f"Logistic GLM (AUC {auc:.3f})", color=INK, fontsize=10, fontweight="bold")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.02)
    pct = FuncFormatter(lambda v, _: f"{v:.0%}")
    a2.xaxis.set_major_formatter(pct)
    a2.yaxis.set_major_formatter(pct)
    a2.set_xlabel("Share of injuries flagged, riskiest first")
    a2.set_ylabel("Share of career endings caught")
    a2.set_title("Career-ending: ranking injuries by risk", pad=10)
    a2.grid(color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    _title(fig, "Both GLMs beat a no-predictor baseline on held-out players",
           f"Scored on the 20% of players not used for fitting: {len(sev.completed(sev_test)):,} completed injuries "
           f"(left; {1 - mae[1] / mae[0]:.0%} lower error) and\n{len(car_test):,} injuries with {int(y.sum())} "
           "career-ending (right).")
    _save(fig, "model_accuracy.png")
    return {"mae": mae, "auc": auc, "top10": top10}


# --------------------------------------------------------------------------
# Pricing back-test (season level, held-out players)
# --------------------------------------------------------------------------
def pricing_figure(bt):
    from runs import backtest
    cal = backtest.by_group(bt, "p_covered_60", "over_60")
    lab = bt[bt["career_labelled"]]
    y = lab["career_ending"].to_numpy()
    top20 = backtest.capture(lab, "p_career", "career_ending")
    pct = FuncFormatter(lambda v, _: f"{v:.0%}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.75, bottom=0.14, wspace=0.3)

    x = cal.index.astype(int).to_numpy()
    a1.bar(x, cal["actual"], width=0.62, color=BASE, label="Actual share")
    a1.plot(x, cal["predicted"], "o-", color=GLM, lw=2, ms=7, mec="white", mew=1.5, label="Model prediction")
    a1.set_xticks(x)
    a1.set_xlabel("Predicted-risk decile (1 = lowest)")
    a1.set_ylabel("Player-seasons with an injury > 60 days")
    a1.yaxis.set_major_formatter(pct)
    a1.set_ylim(0, cal[["predicted", "actual"]].max().max() * 1.18)
    a1.legend(frameon=False, loc="upper left", fontsize=10)
    a1.set_title("Injury beyond a 60-day deferment", pad=10)
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.set_axisbelow(True)

    order = np.argsort(-lab["p_career"].to_numpy())
    flagged = np.r_[0, np.arange(1, len(y) + 1) / len(y)]
    captured = np.r_[0, np.cumsum(y[order]) / y.sum()]
    a2.plot([0, 1], [0, 1], color=BASE, lw=2, ls="--")
    a2.step(flagged, captured, where="post", color=GLM, lw=2)
    a2.plot([0.2], [top20], "o", ms=8, color=GLM, mec="white", mew=2)
    a2.annotate(f"Riskiest 20% of seasons contain\n{int(round(top20 * y.sum()))} of the {int(y.sum())} career endings",
                (0.2, top20), xytext=(0.33, 0.25), fontsize=10, arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1})
    a2.text(0.66, 0.56, "No predictors", color=INK_2, fontsize=9.5)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.02)
    a2.xaxis.set_major_formatter(pct)
    a2.yaxis.set_major_formatter(pct)
    a2.set_xlabel("Share of player-seasons flagged, riskiest first")
    a2.set_ylabel("Share of career endings caught")
    a2.set_title("Career-ending injury in the season", pad=10)
    a2.grid(color=GRID, lw=0.8)
    a2.set_axisbelow(True)

    p60, a60 = bt["p_covered_60"].mean(), bt["over_60"].mean()
    _title(fig, "Season back-test on held-out players: injuries beyond 60 days are calibrated",
           f"{len(bt):,} held-out big-five player-seasons. Left: predicted {p60:.1%} vs actual {a60:.1%} with an "
           f"injury > 60 days. Right: {len(lab):,} seasons with\nknown follow-up; predicted "
           f"{lab['p_career'].mean():.2%} vs actual {y.mean():.2%} career-ending.")
    _save(fig, "pricing_backtest.png")
    return cal.round(3)


def main():
    OUT.mkdir(exist_ok=True)
    print("Severity model ...")
    s_train, s_test = sev.split(sev.build_dataset())
    s_models = sev.fit_all(s_train)
    print(severity_figure(s_models["Gamma GLM, severity tiers"], s_train)[["tier", "mult", "median"]].to_string())
    print("Career-ending model ...")
    c_train, c_test = career.split(career.build_dataset())
    c_models = career.fit_all(c_train)
    print(career_figure(c_models["Logistic GLM, main effects"], c_train).round(2).to_string())
    print(accuracy_figure(s_models, s_test, c_models, c_test))
    print("Pricing back-test ...")
    from runs import backtest
    print(pricing_figure(backtest.run()).to_string())


if __name__ == "__main__":
    main()
