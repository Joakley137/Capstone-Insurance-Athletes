"""
club_premiums.py — what could a club's insurance premium be paying for?

    python -m runs.club_premiums             # reads runs/club_financials.csv, prints the comparison (a few minutes)

Two ways the premium could be priced, both as a share of the club's payroll so clubs of different size compare:

  H1  Wages for injured players while they are out. A player's wage is his share of the payroll, taken
      proportional to his market value on 1 July (the data has no wages). Each injury costs wage x days out / 365.
      Policies usually pay only after a waiting period, so H1 is computed for several deductibles (only the days
      beyond the deductible are paid).
  H2  The rest of a player's contract when an injury ends his career: P(career-ending) x years of contract left
      (not in the data: 1, 2 and 3 are tried).

Each is computed two ways. REALIZED uses what happened to the club's players that season. EXPECTED uses the pricing
models for each squad player (frequency: expected injuries; severity: P(days > D) and the mean days beyond D, from
the tiered gamma GLM averaged over the injury mix; career-ending: the logistic GLM), so it follows each squad's age,
history and league and is smoother than one club-season of luck. The realized-vs-expected comparison is also a check
of the models on these squads.

Then the IMPLIED DEDUCTIBLE (the waiting period at which H1 + H2 equals the premium), the LOADING the premium would
contain at each possible deductible (one premium cannot give both), and two tests of "only some players are insured":
a wage threshold and "the N best-paid players".

Needs the database loaded (python -m database.build_db data/football-datasets) and the Transfermarkt CSVs for club squads
and market values (folder from $FOOTBALL_DATA, default data/football-datasets). Club names are in TEAMS.
"""
import glob
import os

import numpy as np
import pandas as pd

from model import career
from model import frequency as freq
from model import predict
from model import pricing
from runs import backtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FOOTBALL_DATA", "data/football-datasets")
FINANCIALS = os.path.join(HERE, "club_financials.csv")
# club_financials.csv name -> team name in player_performances.csv (first team only)
TEAMS = {"Olympique Lyonnais": "Olympique Lyon", "Real Madrid": "Real Madrid", "AC Milan": "AC Milan",
         "Borussia Dortmund": "Borussia Dortmund"}

DEDUCTIBLES = [0, 28, 60, 90]
YEARS_LEFT = [1, 2, 3]
LOADING_D = (0, 28, 60, 90, 120, 180, 270)               # deductibles shown in the implied-loading table
FIT_D = [0, 14, 28, 45, 60, 90, 120, 180, 270, 365]      # deductibles tried in the insured-player fits
GRID = list(range(0, 366, 14))                           # deductibles tried when solving for the implied one
ALL_D = sorted(set(DEDUCTIBLES) | set(LOADING_D) | set(FIT_D) | set(GRID))
THRESHOLDS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]     # EUR million a year: only players paid at least this are insured
TOP_N = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]        # alternative rule: the N best-paid players of each club
FIT_LOADINGS = [0.0, 0.2, 0.4]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_financials(path=FINANCIALS):
    d = pd.read_csv(path)
    d.columns = ["club", "label", "payroll", "premium"]
    d["season"] = d["label"].str[:4].astype(int)                  # "2023/24" -> 2023
    d["premium_pct"] = d["premium"] / d["payroll"] * 100
    return d


def _csv(name):
    found = sorted(glob.glob(os.path.join(DATA_DIR, "**", f"{name}*.csv"), recursive=True))
    if not found:
        raise FileNotFoundError(f"No {name}*.csv under {DATA_DIR}; set $FOOTBALL_DATA to the football-datasets folder")
    return found[0]


def squads(teams=TEAMS):
    """Players with an appearance for the club's first team, per club and July-June season."""
    raw = pd.read_csv(_csv("player_performances"), usecols=["player_id", "season_name", "team_name", "nb_on_pitch"],
                      low_memory=False)
    raw = raw[raw["team_name"].isin(teams.values()) & (pd.to_numeric(raw["nb_on_pitch"], errors="coerce") > 0)]
    yy = pd.to_numeric(raw["season_name"].astype(str).str.extract(r"^(\d{2})/\d{2}$")[0], errors="coerce")
    out = pd.DataFrame({"club": raw["team_name"].map({v: k for k, v in teams.items()}),
                        "season": yy + np.where(yy > 50, 1900, 2000), "pid": raw["player_id"].astype(str)})
    return out.dropna().astype({"season": int}).drop_duplicates().reset_index(drop=True)


def market_values():
    mv = pd.read_csv(_csv("player_market_value"), usecols=["player_id", "date_unix", "value"], low_memory=False)
    return pd.DataFrame({"pid": mv["player_id"].astype(str),
                         "value_date": pd.to_datetime(mv["date_unix"], errors="coerce").astype("datetime64[ns]"),
                         "market_value": pd.to_numeric(mv["value"], errors="coerce")}).dropna()


def wage_shares(squad, mv):
    """Each squad player's share of the club's payroll, proportional to market value on 1 July.
    A player with no value gets the squad's median; shares sum to 1 within a club-season."""
    s = squad.copy()
    s["t0"] = pd.to_datetime(dict(year=s["season"], month=7, day=1)).astype("datetime64[ns]")
    s = s.sort_values("t0")
    hit = pd.merge_asof(s.reset_index(), mv[mv["market_value"] > 0].sort_values("value_date"), left_on="t0",
                        right_on="value_date", by="pid", direction="backward",
                        tolerance=pd.to_timedelta(730, unit="D")).set_index("index")
    s["value"] = hit["market_value"]
    key = ["club", "season"]
    s["value"] = s["value"].fillna(s.groupby(key)["value"].transform("median")).fillna(1.0)
    s["share"] = s["value"] / s.groupby(key)["value"].transform("sum")
    return s.drop(columns="t0").sort_index()


# --------------------------------------------------------------------------
# Realized loss
# --------------------------------------------------------------------------
def realized_h1(squad, inj, deductibles=DEDUCTIBLES):
    """Share of payroll paid out under H1, per club-season and deductible. inj: freq.covered_injuries()
    (pid, season, days_missed). An injury counts in the season it started and costs the player's payroll share
    x (days beyond the deductible) / 365."""
    m = squad.merge(inj[["pid", "season", "days_missed"]], on=["pid", "season"], how="left")
    out = squad.groupby(["club", "season"]).size().rename("players").to_frame()
    for d in deductibles:
        covered = (m["days_missed"] - d).clip(lower=0).fillna(0)
        out[f"h1_d{d}_pct"] = (m["share"] * covered / 365).groupby([m["club"], m["season"]]).sum() * 100
    return out.reset_index()


def career_ending_rate(fd):
    """Chance a big-five player's career ends in a season, per player-season, with career.py's definition
    (no professional appearance after the injury and not back by season end). Only labelled seasons count."""
    act = backtest.actuals(fd, career.build_dataset())
    lab = act["career_labelled"]
    return act.loc[lab, "career_ending"].mean(), int(act.loc[lab, "career_ending"].sum()), int(lab.sum())


def compare(fin, h1, rate, years_left=YEARS_LEFT):
    t = fin.merge(h1, on=["club", "season"])
    for y in years_left:
        t[f"h2_y{y}_pct"] = rate * y * 100
    return t


def summary(t, deductibles=DEDUCTIBLES, years_left=YEARS_LEFT):
    """Per club: average premium % of payroll and what each hypothesis implies, as a multiple of the premium."""
    cols = {f"h1_d{d}_pct": f"H1, {d}-day deductible" for d in deductibles}
    cols.update({f"h2_y{y}_pct": f"H2, {y} yr left" for y in years_left})
    g = t.groupby("club")
    out = pd.DataFrame({"seasons": g.size(), "premium_pct": g["premium_pct"].mean()})
    for c, name in cols.items():
        out[name + " (x premium)"] = g[c].mean() / out["premium_pct"]
    return out.round(2)


# --------------------------------------------------------------------------
# Expected loss from the pricing models
# --------------------------------------------------------------------------
def mix_cells(b, inputs):
    """One row per squad player-season and injury-mix cell (region x type): the features the severity and career
    models need, as backtest.mix_probs builds them. Returns (rows, mix weights w, n players, k cells)."""
    mix = b["mix"]
    n, k = len(inputs), len(mix)
    rows = inputs.loc[inputs.index.repeat(k)].reset_index(drop=True)
    rows["region"] = np.tile(mix["region"].to_numpy(), n)
    rows["nature"] = np.tile(mix["nature"].to_numpy(), n)
    cls = [predict._rating_class(b, r, t) for r, t in zip(mix["region"], mix["nature"])]
    rows["injury_class"] = np.tile(cls, n)
    return rows, mix["w"].to_numpy(), n, k


def excess_days(b, rows, w, n, k, deductibles):
    """Expected days beyond each deductible for a random injury of each player: sum_k w_k P(days > d) (E[days | days > d] - d),
    from the tiered gamma GLM and its per-tier spread."""
    out = {}
    for d in deductibles:
        sf, cond = pricing._tail(b["tiered"], rows, d)
        out[d] = (sf * (cond - d)).reshape(n, k) @ w
    return out


def squad_rows(squad, fd, b):
    """Squad player-seasons joined to the frequency data (age, position, league, history) with the expected number
    of injuries lam. Players missing from the frequency data are left out and the shares renormalised
    (matched_share is the share of squad value that was kept)."""
    rows = squad.merge(fd, on=["pid", "season"], how="inner")
    rows["matched_share"] = rows.groupby(["club", "season"])["share"].transform("sum")
    rows["share"] = rows["share"] / rows["matched_share"]
    rows = rows.reset_index(drop=True)
    rows["lam"] = freq.expected_injuries(b["frequency"], rows)
    return rows


def squad_inputs(squad, fd, b, deductibles=ALL_D):
    """Per squad player-season: payroll share, expected injuries (lam), P(an injury ends his career) (c) and the
    expected days beyond each deductible per injury (ex_d{d})."""
    rows = squad_rows(squad, fd, b)
    cells, w, n, k = mix_cells(b, backtest.season_inputs(rows))
    out = rows[["club", "season", "pid", "share", "matched_share", "lam"]].copy()
    out["c"] = pricing._career_p(b, cells).reshape(n, k) @ w
    for d, v in excess_days(b, cells, w, n, k, deductibles).items():
        out[f"ex_d{d}"] = v
    return out


def player_losses(p, deductibles=ALL_D, years_left=YEARS_LEFT):
    """Expected loss per squad player-season, in % of the club's payroll: w = payroll share x expected injuries.
    H1: w x (expected days beyond the deductible) / 365.  H2: w x P(career-ending | injury) x years of contract left."""
    w = p["share"] * p["lam"]
    out = p[["club", "season", "share"]].copy()
    for d in deductibles:
        out[f"h1_d{d}"] = w * p[f"ex_d{d}"] / 365 * 100
    for y in years_left:
        out[f"h2_y{y}"] = w * p["c"] * y * 100
    return out


def expected_losses(p, deductibles=ALL_D, years_left=YEARS_LEFT):
    pl = player_losses(p, deductibles, years_left).drop(columns="share")
    out = pl.groupby(["club", "season"]).sum().reset_index()
    out.columns = [c if c in ("club", "season") else f"e_{c}_pct" for c in out.columns]
    return out.merge(p.groupby(["club", "season"])["matched_share"].first().reset_index(), on=["club", "season"])


def implied_loading(pct_by_d, premium_pct, h2_pct, deductibles=LOADING_D):
    """If the policy had deductible d, the loading the premium would have to contain: premium / expected payout - 1.
    Negative means the premium is below the expected payout (so that deductible is too short)."""
    return {d: premium_pct / (pct_by_d[d] + h2_pct) - 1 for d in deductibles}


def implied_deductible(pct_by_d, premium_pct, h2_pct):
    """The deductible (days) at which H1 + H2 equals the premium. pct_by_d: H1 (% of payroll) indexed by
    deductible, falling as the deductible rises. 0 means the premium covers everything from day one;
    inf means even a one-year deductible leaves H1 + H2 above the premium; nan means H2 alone exceeds it."""
    target = premium_pct - h2_pct
    if target <= 0:
        return np.nan
    d = np.asarray(pct_by_d.index, float)
    v = np.asarray(pct_by_d.values, float)
    if v[0] <= target:
        return 0.0
    if v[-1] > target:
        return np.inf
    return float(np.interp(target, v[::-1], d[::-1]))


# --------------------------------------------------------------------------
# Only some players insured?
# --------------------------------------------------------------------------
def insured_premium(pl, payroll, threshold, deductible, years_left, loading, top_n=None):
    """Predicted premium (% of payroll) per club-season if only players with a wage of at least `threshold` (EUR m)
    -- or only the top_n best-paid -- are insured. A player's wage is his payroll share x the club's payroll."""
    m = pl.merge(payroll, on=["club", "season"])
    insured = (m["share"] * m["payroll"]) >= threshold
    if top_n is not None:
        insured = m.groupby(["club", "season"])["share"].rank(ascending=False, method="first") <= top_n
    m = m.assign(n=insured.astype(int), cost=np.where(insured, m[f"h1_d{deductible}"] + m[f"h2_y{years_left}"], 0.0))
    g = m.groupby(["club", "season"]).agg(cost=("cost", "sum"), n_insured=("n", "sum"))
    g["predicted_pct"] = (1 + loading) * g["cost"]
    return g.reset_index()


def _fit(pl, fin, rule, values, years_left, deductibles, loadings):
    payroll = fin[["club", "season", "payroll"]]
    actual = fin.groupby("club")["premium_pct"].mean()
    rows = []
    for v in values:
        for d in deductibles:
            for th in loadings:
                g = (insured_premium(pl, payroll, v, d, years_left, th) if rule == "threshold_m"
                     else insured_premium(pl, payroll, 0, d, years_left, th, top_n=v))
                club = g.groupby("club")[["predicted_pct", "n_insured"]].mean()
                err = np.mean(np.log(club["predicted_pct"].clip(lower=1e-4) / actual.reindex(club.index)) ** 2)
                rows.append({"loading": th, rule: v, "deductible": d, "error": err,
                             **{f"pred_{c}": club.loc[c, "predicted_pct"] for c in club.index},
                             **{f"n_{c}": club.loc[c, "n_insured"] for c in club.index}})
    return pd.DataFrame(rows)


def fit_threshold(pl, fin, years_left=2, thresholds=THRESHOLDS, deductibles=FIT_D, loadings=FIT_LOADINGS):
    """For each loading, the (wage threshold, deductible) that best reproduces the clubs' average premiums, as the
    mean squared log error across clubs. Threshold 0 (everyone insured) is in the grid for comparison."""
    return _fit(pl, fin, "threshold_m", thresholds, years_left, deductibles, loadings)


def fit_top_n(pl, fin, years_left=2, top_ns=TOP_N, deductibles=FIT_D, loadings=FIT_LOADINGS):
    """Same search with the rule "the N best-paid players of each club are insured"."""
    return _fit(pl, fin, "top_n", top_ns, years_left, deductibles, loadings)


# --------------------------------------------------------------------------
def load_base():
    """Financials, the frequency data, the squads with payroll shares and the fitted pricing models."""
    fd = freq.build_dataset()
    last = int(fd["season"].max())
    fin = load_financials()
    fin = fin[fin["season"].between(freq.FIRST_SEASON, last)]
    squad = wage_shares(squads(), market_values())
    squad = squad[squad["season"].between(freq.FIRST_SEASON, last)].reset_index(drop=True)
    return fin, fd, squad, pricing.models()


def load_inputs():
    """load_base() plus the expected-loss inputs for every squad player-season."""
    fin, fd, squad, b = load_base()
    return fin, fd, squad, b, squad_inputs(squad, fd, b)


def main():
    pd.set_option("display.width", 220)
    fin, fd, squad, b, p = load_inputs()

    # --- realized ---
    inj = freq.covered_injuries()
    grid_h1 = realized_h1(squad, inj, GRID)
    h1 = realized_h1(squad, inj, DEDUCTIBLES)
    rate, n_events, n_seasons = career_ending_rate(fd)
    print(f"career-ending: {n_events} events in {n_seasons:,} big-five player-seasons = {rate:.3%} per season")
    t = compare(fin, h1, rate)
    print("\nREALIZED, per club, as a multiple of the premium (1.0 = matches):")
    print(summary(t).T.to_string())

    # --- expected, from the models ---
    e = expected_losses(p)
    te = fin.merge(e, on=["club", "season"]).merge(h1, on=["club", "season"])
    print(f"\nSquad value matched to the frequency data: {te['matched_share'].mean():.0%} on average")
    check = te.groupby("club").apply(lambda g: pd.Series({
        "expected_h1_d0_pct": g["e_h1_d0_pct"].mean(), "realized_h1_d0_pct": g["h1_d0_pct"].mean(),
        "expected_h1_d90_pct": g["e_h1_d90_pct"].mean(), "realized_h1_d90_pct": g["h1_d90_pct"].mean(),
        "corr_d0 (seasons)": g["e_h1_d0_pct"].corr(g["h1_d0_pct"])}), include_groups=False)
    print("\nModel check: expected vs realized loss on these squads, % of payroll")
    print(check.round(2).to_string())

    cols = {f"e_h1_d{d}_pct": f"expected H1, {d}-day deductible" for d in DEDUCTIBLES}
    cols.update({f"e_h2_y{y}_pct": f"expected H2, {y} yr left" for y in YEARS_LEFT})
    g = te.groupby("club")
    out = pd.DataFrame({"seasons": g.size(), "premium_pct": g["premium_pct"].mean()})
    for c, name in cols.items():
        out[name + " (x premium)"] = g[c].mean() / out["premium_pct"]
    print("\nEXPECTED, per club, as a multiple of the premium (1.0 = matches):")
    print(out.round(2).T.to_string())

    # --- implied deductible ---
    rows = []
    for club, gg in te.groupby("club"):
        r = {"club": club, "premium_pct": gg["premium_pct"].mean()}
        real_curve = grid_h1[grid_h1["club"] == club].set_index("season").loc[gg["season"]].filter(like="h1_d").mean()
        real_curve.index = [int(k.split("_d")[1].split("_")[0]) for k in real_curve.index]
        exp_curve = pd.Series({d: gg[f"e_h1_d{d}_pct"].mean() for d in GRID})
        for y in YEARS_LEFT:
            r[f"expected, {y} yr left"] = implied_deductible(exp_curve, r["premium_pct"], gg[f"e_h2_y{y}_pct"].mean())
            r[f"realized, {y} yr left"] = implied_deductible(real_curve.reindex(GRID), r["premium_pct"], rate * y * 100)
        rows.append(r)
    print("\nIMPLIED DEDUCTIBLE (days): the waiting period at which H1 + H2 equals the average premium")
    print("(0 = covers every day; inf = even a 365-day deductible leaves the losses above the premium; nan = H2 alone exceeds it)")
    imp = pd.DataFrame(rows).set_index("club")
    print(imp.round({c: 0 for c in imp.columns if c != "premium_pct"}).round({"premium_pct": 2}).to_string())

    print("\nLOADING the premium would contain if the policy had this deductible (model-based expected payout, 2 yr left;")
    print("0% = the premium equals the expected payout; negative = the premium is below the expected payout):")
    lrows = []
    for club, gg in te.groupby("club"):
        curve = {d: gg[f"e_h1_d{d}_pct"].mean() for d in LOADING_D}
        lrows.append({"club": club, **implied_loading(curve, gg["premium_pct"].mean(), gg["e_h2_y2_pct"].mean())})
    print((pd.DataFrame(lrows).set_index("club") * 100).round(0).rename(columns=lambda d: f"{d}-day").to_string())

    # --- only some players insured? ---
    pl = player_losses(p, FIT_D, [2])
    fits = fit_threshold(pl, fin)
    print("\nONLY THE BEST-PAID PLAYERS INSURED? Best (wage threshold in EUR m a year, deductible in days) per loading,")
    print("chosen so one pair reproduces all clubs' average premiums. Threshold 0 = everyone insured:")
    for th, gg in fits.groupby("loading"):
        best, flat = gg.loc[gg["error"].idxmin()], gg[gg["threshold_m"] == 0].loc[lambda x: x["error"].idxmin()]
        print(f"  loading {th:.0%}: threshold {best.threshold_m:g}m, deductible {best.deductible:g} days, error {best.error:.3f}"
              f"   |  no threshold: deductible {flat.deductible:g} days, error {flat.error:.3f}")
    tn = fit_top_n(pl, fin)
    print("\nALTERNATIVE: the N best-paid players of each club are insured. Best (N, deductible) per loading:")
    for th, gg in tn.groupby("loading"):
        best = gg.loc[gg["error"].idxmin()]
        print(f"  loading {th:.0%}: top {best.top_n:g} players, deductible {best.deductible:g} days, error {best.error:.3f}   |   "
              + ", ".join(f"{c[5:]} {best[c]:.2f}" for c in tn.columns if c.startswith("pred_")))


if __name__ == "__main__":
    main()
