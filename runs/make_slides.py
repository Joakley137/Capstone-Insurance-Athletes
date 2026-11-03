"""
make_slides.py — one 16:9 slide per GLM: the fitted equation, the multipliers exp(coef) with 95% intervals,
and the facts needed to read them.

    python -m runs.make_slides         # writes figures/slides/frequency.png, severity.png, career.png

Coefficients come from the all-data fits that `python -m model.predict` uses (same as MODEL_EQUATIONS.md).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, NullFormatter

from model import career, frequency, predict
from model import severity as sev

OUT = Path(__file__).parent / "figures" / "slides"
GLM, BASE = "#2a78d6", "#a3a29c"                       # significant (p < 0.05) blue, not significant gray
INK, INK_2, GRID, PANEL = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3f0"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 13, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "mathtext.fontset": "dejavusans",
})

LEAGUE = {"ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}
POSITION = {"Goalkeeper": "Goalkeeper", "Defender": "Defender", "Attack": "Attacker"}
CAREER_EVENTS = 504                                    # career-ending injuries in the fit (MODEL_EQUATIONS.md); saved models drop y
TIER = {1: "illness", 2: "knocks, head", 3: "foot/knee knocks, concussion", 4: "ankle ligament, strains",
        5: "calf, groin, head fracture", 6: "ankle, hamstring strain, hip", 8: "knee, shoulder",
        9: "ankle/thigh tear, knee tendon", 10: "foot fracture, groin surgery", 11: "knee ligament, other surgery",
        12: "ankle surgery, leg fracture", 13: "ACL, Achilles tear, knee surgery"}
SHOW = {                                               # terms plotted per slide, top to bottom (names from sev.coefficients)
    "frequency": ["log(1 + prior_injuries)", "league = L1", "league = IT1", "log(1 + games_last_season)",
                  "log(1 + days_out_last_season)", "league = FR1", "position_group = Goalkeeper"],
    "severity": ["severity_tier = Tier 13", "severity_tier = Tier 11", "prior_same_part", "reinjury_60d",
                 "log(1 + prior_injuries)", "league = IT1", "severity_tier = Tier 1"],
    "career": ["region_g = Knee", "region_g = Achilles", "nature_g = Surgery", "nature_g = Tear / rupture", "age_c",
               "log(1 + prior_injuries)", "log(1 + games_12m)"],
}


def _table(model):
    """coefficients() plus 95% interval on the multiplier scale, intercept dropped."""
    t = sev.coefficients(model)
    se = np.asarray(model.res.bse)
    t["mult"], t["lo"], t["hi"] = np.exp(t["coef"]), np.exp(t["coef"] - 1.96 * se), np.exp(t["coef"] + 1.96 * se)
    return t


def _rows(t, labels, name):
    """Rows for the terms in SHOW[name], in that order; labels must cover every term so any selection works."""
    terms = [k for k in t.index if k != "Intercept"]
    assert set(terms) == set(labels), set(terms) ^ set(labels)
    return [(labels[k], t.loc[k]) for k in SHOW[name]], len(terms)


def _forest(ax, rows, xlabel):
    n = len(rows)
    for i, (label, r) in enumerate(rows):
        y = n - 1 - i
        sig = r["p_value"] < 0.05
        c = GLM if sig else BASE
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=3.2, solid_capstyle="round")
        ax.plot(r["mult"], y, "o", color=c, ms=11, mfc=c if sig else "white", mew=2.2)
        note = f"{r['mult']:.2f}×" + ("" if sig else f"  (p = {r['p_value']:.2f})")
        ax.text(1.03, y, note, transform=ax.get_yaxis_transform(), va="center", fontsize=14, color=c,
                fontweight="bold" if sig else "normal")
        label = label.replace(" (vs", "\n(vs") if len(label) > 34 else label
        ax.text(-0.02, y, label, linespacing=1.15, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=15)
    ax.axvline(1, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.set_xscale("log")
    lo, hi = min(min(r["lo"] for _, r in rows), 1), max(max(r["hi"] for _, r in rows), 1)
    pad = (hi / lo) ** 0.08
    xlim = (lo / pad, hi * pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    span = np.log(xlim[1] / xlim[0])                   # finest tick set that covers both ends without crowding
    for steps in ([0.5, 0.6, 0.8, 1, 1.25, 1.6, 2], [0.3, 0.5, 0.7, 1, 1.5, 2, 3], [0.25, 0.5, 1, 2, 4, 8],
                  [0.1, 0.2, 0.5, 1, 3, 10, 30, 100], [0.01, 0.1, 1, 10, 100]):
        ticks = [t for t in steps if xlim[0] <= t <= xlim[1]]
        if (np.diff(np.log(ticks)).min() >= 0.14 * span and ticks[0] <= 2 * xlim[0] and ticks[-1] >= xlim[1] / 2):
            break
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=12.5)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=12.5)


def _slide(name, title, takeaway, equation, eq_size, rows, xlabel, facts, eq2=None):
    rows, m = rows
    fig = plt.figure(figsize=(13.333, 7.5))
    fig.text(0.04, 0.945, title, fontsize=26, fontweight="bold", va="top")
    fig.text(0.04, 0.875, takeaway, fontsize=14.5, color=INK_2, va="top")

    band_h = 0.13 if eq2 else 0.095
    band_y = 0.845 - band_h
    fig.patches.append(FancyBboxPatch((0.04, band_y), 0.92, band_h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=PANEL, ec="none"))
    if eq2:
        fig.text(0.5, band_y + band_h * 0.70, equation, fontsize=eq_size, ha="center", va="center")
        fig.text(0.5, band_y + band_h * 0.27, eq2, fontsize=eq_size, ha="center", va="center")
    else:
        fig.text(0.5, band_y + band_h / 2, equation, fontsize=eq_size, ha="center", va="center")

    top = band_y - 0.035
    ax = fig.add_axes([0.37, 0.15, 0.27, top - 0.15])
    _forest(ax, rows, xlabel)

    x0 = 0.75
    fig.patches.append(FancyBboxPatch((x0 - 0.02, 0.13), 0.23, top - 0.13, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=PANEL, ec="none"))
    y = top - 0.03
    for head, body in facts:
        fig.text(x0, y, head.upper(), fontsize=10, fontweight="bold", color=INK_2, va="top")
        fig.text(x0, y - 0.032, body, fontsize=12, va="top", linespacing=1.35)
        y -= 0.032 + 0.031 * (body.count("\n") + 1) + 0.035

    fig.text(0.04, 0.025, "● p < 0.05      ○ not significant      bars: 95% interval      dashed line: no effect (1×)",
             fontsize=10.5, color=INK_2)
    fig.text(0.96, 0.025, f"Showing {len(rows)} of the {m} fitted terms; full table in MODEL_EQUATIONS.md",
             fontsize=10.5, color=INK_2, ha="right")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/slides/{name}.png")


def frequency_slide(b):
    m = b["frequency"]
    t = _table(m)
    rows = _rows(t, {
        **{f"league = {k}": f"{v} (vs Premier League)" for k, v in LEAGUE.items()},
        **{f"position_group = {k}": f"{v} (vs midfielder)" for k, v in POSITION.items()},
        "age_c": "Age, per year from 26", "I(age_c ** 2)": "Age² (curvature)",
        "log(1 + games_last_season)": "ln(1 + games last season)", "no_games_last_season": "No minutes last season",
        "log(1 + prior_injuries)": "ln(1 + prior injuries)", "log(1 + days_out_last_season)": "ln(1 + days out last season)",
    }, "frequency")
    _slide("frequency", "Frequency: how many injuries per season",
           "Negative binomial GLM. Injury history and playing load raise the rate; goalkeepers are injured 35% less often.",
           r"$\ln\lambda = \beta_0 + \beta_{\mathrm{league}} + \beta_{\mathrm{position}} + \beta_1 a + \beta_2 a^2"
           r" + \beta_3 \ln(1+g) + \beta_4 Z + \beta_5 \ln(1+n) + \beta_6 \ln(1+d)$", 19.5,
           rows, "Rate ratio exp(β): multiplier on expected injuries",
           [("Outcome", "N = injuries in a season\nN ~ NegBin(λ),  α = %.3f" % m.alpha),
            ("Fitted on", f"{b['n_player_seasons']:,} big-five player-seasons\n2015/16 to 2024/25"),
            ("Reference player", "Premier League midfielder, 26\nexp(β₀) = %.3f" % np.exp(t.loc["Intercept", "coef"])),
            ("Example", "27-year-old PL defender,\n2,500 min, 4 prior injuries\nλ = 1.00 injuries, P(any) = 59%")])


def severity_slide(b):
    t = _table(b["tiered"])
    rows = _rows(t, {
        **{f"severity_tier = Tier {k}": f"Tier {k}: {v} (vs Tier 7)" for k, v in TIER.items()},
        **{f"league = {k}": f"{v} (vs Premier League)" for k, v in LEAGUE.items()},
        **{f"pos_{k.lower()}": f"{v} (vs midfielder)" for k, v in POSITION.items()},
        "age_c": "Age, per year from 26", "I(age_c ** 2)": "Age² (curvature)",
        "log(1 + prior_injuries)": "ln(1 + prior injuries)",
        "reinjury_60d": "Within 60 days of return", "prior_same_part": "Same body part before",
    }, "severity")
    _slide("severity", "Severity: how long an injury lasts",
           "Gamma GLM. The injury tier sets the length: a Tier 13 injury (mostly ACL) lasts 4.7× the reference, illness 0.3×.",
           r"$\ln\mu = \beta_0 + \beta_{\mathrm{tier}} + \beta_{\mathrm{league}} + \beta_{\mathrm{position}}"
           r" + \beta_1 a + \beta_2 a^2 + \beta_3 \ln(1+n) + \beta_4 R + \beta_5 S$", 22,
           rows, "exp(β): multiplier on expected days out",
           [("Outcome", "μ = expected days out\ndays ~ Gamma(μ), log link"),
            ("Fitted on", f"{int(b['tiered'].res.nobs):,} injuries in\nbig-five seasons"),
            ("Reference injury", "Tier 7 (e.g. hamstring),\nPL midfielder, 26\nexp(β₀) = %.1f days" % np.exp(t.loc["Intercept", "coef"])),
            ("Used for", "q = P(days > 60), the chance\nan injury is covered")])


def career_slide(b):
    t = _table(b["career"])
    rows = _rows(t, {
        **{f"region_g = {k}": f"{v} (vs hamstring)" for k, v in
           {"Achilles": "Achilles", "Knee": "Knee", "Unknown": "Body part not recorded", "Other": "Other body part"}.items()},
        **{f"nature_g = {k}": f"{v} (vs unspecified type)" for k, v in
           {"Surgery": "Surgery", "Tear / rupture": "Tear / rupture", "Fracture": "Fracture",
            "Ligament / joint": "Ligament / joint", "Other": "Other type"}.items()},
        **{f"position_group = {k}": f"{v} (vs midfielder)" for k, v in POSITION.items()},
        "age_c": "Age, per year from 26", "I(age_c ** 2)": "Age² (curvature)",
        "log(1 + prior_injuries)": "ln(1 + prior injuries)",
        "reinjury_60d": "Within 60 days of return", "prior_same_part": "Same body part before",
        "log(1 + games_12m)": "ln(1 + games, last 12 months)", "no_games_12m": "No games, last 12 months",
        "log(1 + career_games)": "ln(1 + career games)", "year_c": "Year, from 2015",
    }, "career")
    _slide("career", "Career-ending: will this injury end the career",
           "Logistic GLM. Knee and Achilles injuries carry about 11× the odds; older and less-used players are at higher risk.",
           r"$\ln\frac{p}{1-p} = \beta_0 + \beta_{\mathrm{region}} + \beta_{\mathrm{type}} + \beta_{\mathrm{position}}"
           r" + \beta_1 a + \beta_2 a^2 + \beta_3 \ln(1+n)$", 21,
           rows, "Odds ratio exp(β): multiplier on the odds of a career-ending injury",
           [("Outcome", "p = P(injury ends career)"),
            ("Fitted on", f"{int(b['career'].res.nobs):,} injuries, all leagues\n{CAREER_EVENTS} career-ending"),
            ("Reference injury", "Unspecified hamstring,\nmidfielder, 26, year 2015\nexp(β₀) = %.4f" % np.exp(t.loc["Intercept", "coef"])),
            ("Used for", "c = P(career-ending), averaged\nover the injury mix")],
           eq2=r"$+\ \beta_4 R + \beta_5 S + \beta_6 \ln(1+g_{12}) + \beta_7 Z_{12} + \beta_8 \ln(1+G) + \beta_9 t$")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = predict.models()
    frequency_slide(b)
    severity_slide(b)
    career_slide(b)


if __name__ == "__main__":
    main()
