"""
premium_test.py — what could a club's insurance premium be paying for?

    python premium_test.py                    # reads club_financials.csv, prints the comparison

Two ways the premium could be priced, both expressed as a share of the club's payroll so clubs of different size compare:

  H1  Wages for injured players while they are out. A player's wage is his share of the payroll, taken
      proportional to his market value on 1 July. Each injury costs wage x days out / 365. Policies usually
      pay only after a waiting period, so H1 is computed for several deductibles (0, 28, 60, 90 days: only days
      beyond the deductible are paid).
  H2  The rest of a player's contract when an injury ends his career. Rare: the league-wide chance a player's
      career ends in a season, times the years of contract left (not in the data: 1, 2 and 3 are tried).

For every club-season it prints the premium as a % of payroll next to what each hypothesis implies. A hypothesis
whose number is many times the premium cannot be what the premium pays for. H1 is REALIZED (what happened to the
club's actual players), so it is noisy for one club-season; the per-club averages are the evidence. H2 is an
EXPECTED amount from the league-wide career-ending rate, because one club has too few career-ending injuries to
measure its own.

Both are computed two ways. REALIZED uses what actually happened to the club's players that season. EXPECTED
uses the frequency model (injuries per player) and the severity model (distribution of days out, so the part beyond
a deductible can be priced) and the career-ending model, for each player on the squad with his own age, history and
value; it is smoother, and the realized-vs-expected comparison also checks the models on these squads.
Then the IMPLIED DEDUCTIBLE (the waiting period at which H1 plus H2 equals the premium) and the LOADING the premium
would contain at each possible deductible. One premium cannot say both, so both views are shown.

Needs the soccer tables loaded, the frequency.py dataset, and club_financials.csv (club, season, payroll, premium).
"""
import numpy as np
import pandas as pd
from scipy import stats

import career
import frequency as freq
import severity as sev
import sportsdb as sdb

DEDUCTIBLES = [0, 28, 60, 90]
YEARS_LEFT = [1, 2, 3]
THRESHOLDS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]    # EUR million gross a year: only players paid at least this are insured
FIT_D = [0, 14, 28, 45, 60, 90, 120, 180, 270, 365]          # deductibles tried in the threshold fit
FIT_LOADINGS = [0.0, 0.2, 0.4]
TOP_N = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]    # alternative rule: insure the N best-paid players of each club
LOADING_D = (0, 28, 60, 90, 120, 180, 270)     # deductibles shown in the implied-loading table
GRID = list(range(0, 366, 7))              # deductibles tried when solving for the implied one
MIX_SIZE = 150                             # real injuries sampled to stand for "which injury it will be"
PLAYER_COLS = ["pos_attack", "pos_defender", "pos_goalkeeper", "position_group", "age_c", "prior_injuries",
               "log_mv_c", "mv_missing", "year_c", "career_games"]
INJURY_COLS = ["injury_class", "region", "nature", "body_part", "prior_same_part", "reinjury_60d", "offseason"]
# club_financials.csv name -> team name in the appearance data (first team only)
TEAMS = {"Olympique Lyonnais": "Olympique Lyon", "Real Madrid": "Real Madrid", "AC Milan": "AC Milan",
         "Borussia Dortmund": "Borussia Dortmund"}


def load_financials(path="club_financials.csv"):
    d = pd.read_csv(path)
    d.columns = ["club", "season", "payroll", "premium"]
    d["wy"] = d["season"].str[:4].astype(int)                     # "2023/24" -> 2023
    d["premium_pct"] = d["premium"] / d["payroll"] * 100
    return d


def squads(teams=TEAMS):
    """Players with an appearance for the club's first team, per club and July-June season."""
    marks = ",".join(f":t{i}" for i in range(len(teams)))
    ps = sdb.query(f"""SELECT ext_player_id AS pid, team, season_start, season_end FROM player_seasons
                       WHERE sport = 'Soccer' AND team IN ({marks}) AND appearances > 0 AND season_end IS NOT NULL""",
                   {f"t{i}": t for i, t in enumerate(teams.values())})
    for c in ["season_start", "season_end"]:
        ps[c] = pd.to_datetime(ps[c])
    ps["wy"] = freq.season_year(ps["season_start"] + (ps["season_end"] - ps["season_start"]) / 2).to_numpy()
    back = {v: k for k, v in teams.items()}
    ps["club"] = ps["team"].map(back)
    return ps[["club", "wy", "pid"]].drop_duplicates().reset_index(drop=True)


def wage_shares(squad, mv):
    """Each squad player's share of the club's payroll, proportional to market value on 1 July.
    A player with no value gets the squad's median; shares sum to 1 within a club-season."""
    s = squad.copy()
    s["t0"] = freq._july(s["wy"]).astype("datetime64[ns]")
    mv = mv.assign(value_date=pd.to_datetime(mv["value_date"]).astype("datetime64[ns]")).rename(
        columns={"ext_player_id": "pid"})
    s = s.sort_values("t0")
    hit = pd.merge_asof(s.reset_index(), mv.sort_values("value_date"), left_on="t0", right_on="value_date",
                        by="pid", direction="backward", tolerance=pd.to_timedelta(730, unit="D")).set_index("index")
    s["value"] = hit["market_value"]
    key = ["club", "wy"]
    s["value"] = s["value"].fillna(s.groupby(key)["value"].transform("median")).fillna(1.0)
    s["share"] = s["value"] / s.groupby(key)["value"].transform("sum")
    return s.drop(columns="t0").sort_index()


def realized_h1(squad, inj, deductibles=DEDUCTIBLES):
    """Share of payroll paid out under H1, per club-season and deductible.

    inj: injuries with pid, injury_date, days_missed. An injury counts in the season it started and costs
    the player's payroll share x (days beyond the deductible) / 365.
    """
    i = inj.assign(wy=freq.season_year(inj["injury_date"]).to_numpy())[["pid", "wy", "days_missed"]]
    m = squad.merge(i, on=["pid", "wy"], how="left")
    out = squad.groupby(["club", "wy"]).size().rename("players").to_frame()
    for d in deductibles:
        covered = (m["days_missed"] - d).clip(lower=0).fillna(0)
        out[f"h1_d{d}_pct"] = (m["share"] * covered / 365).groupby([m["club"], m["wy"]]).sum() * 100
    return out.reset_index()


def career_ending_rate(max_year):
    """Chance a top-5-league player's career ends in a season (League-wide, per player-season), using career.py's
    definition: no professional appearance after the injury and not back by season end. Also counts at the four clubs."""
    ce = career.build_dataset()
    ce = ce[ce["career_ending"] == 1]
    ce = ce.assign(wy=freq.season_year(ce["injury_date"]).to_numpy(), pid=ce["ext_player_id"])
    data = freq.build_dataset(start_year=freq.DEFAULT_START_YEAR, leagues=freq.TOP5_LEAGUES)
    data = data[data["wy"] <= max_year]
    events = ce.merge(data[["pid", "wy"]], on=["pid", "wy"])
    return len(events) / len(data), len(events), len(data)


def compare(fin, h1, rate, years_left=YEARS_LEFT):
    t = fin.merge(h1, on=["club", "wy"])
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
# Expected loss from the models
# --------------------------------------------------------------------------
def lognormal_excess(mu, sigma, deductibles):
    """E[(X - d)+] for a lognormal with log-mean mu (array) and log-sd sigma: the expected days beyond a
    deductible of d days. d = 0 gives the mean."""
    mean = np.exp(mu + sigma ** 2 / 2)
    out = {}
    for d in deductibles:
        if d == 0:
            out[d] = mean
        else:
            out[d] = mean * stats.norm.cdf((mu + sigma ** 2 - np.log(d)) / sigma) - d * stats.norm.cdf((mu - np.log(d)) / sigma)
    return out


def cross_join(players, mix):
    """Every player row paired with every sampled injury (player columns repeated, injury columns tiled)."""
    n, m = len(players), len(mix)
    p = players.loc[players.index.repeat(m)].reset_index(drop=True)
    x = pd.concat([mix.reset_index(drop=True)] * n, ignore_index=True)
    return pd.concat([p, x], axis=1)


def squad_inputs(squad, freq_data, poisson, lognormal, career_model, sev_data, mix_size=MIX_SIZE, seed=0):
    """Per squad player: payroll share, expected injuries, and for each of mix_size sampled real injuries the
    lognormal log-mean of the days out and the chance the injury ends his career.

    The injury mix is a random sample of real injuries (so class, body part and recurrence flags keep their real
    mix); the player's own age, history, position, value and minutes come from his row in the frequency data.
    Players missing from the frequency data are left out and the shares renormalised.
    Returns (players frame with club, wy, share, lam, matched_share), mu (players x mix), pce (players x mix).
    """
    cols = ["pid", "wy", "games_prior", "no_games_prior", "prior_last_year", "prior_days_out_1y", "prior_major",
            "carry_in"] + PLAYER_COLS
    p = squad.merge(freq_data[cols], on=["pid", "wy"], how="inner")
    p["matched_share"] = p.groupby(["club", "wy"])["share"].transform("sum")
    p["share"] = p["share"] / p["matched_share"]
    p = p.reset_index(drop=True)
    p["lam"] = poisson.rate(p)
    p["games_12m"] = p["games_prior"]
    p["no_games_12m"] = p["no_games_prior"]
    p["year"] = p["year_c"] + 2015
    mix = sev_data[INJURY_COLS].sample(mix_size, random_state=seed)
    rows = cross_join(p.drop(columns=["share", "lam", "matched_share"]), mix)
    n, m = len(p), len(mix)
    mu = lognormal._xb(rows).reshape(n, m)
    pce = np.asarray(career_model.predict(rows)).reshape(n, m)
    return p[["club", "wy", "share", "lam", "matched_share"]], mu, pce


def expected_losses(p, mu, pce, sigma, deductibles=DEDUCTIBLES, years_left=YEARS_LEFT):
    """Expected share of payroll lost per club-season, from squad_inputs().

    Per squad player, with w = payroll share x expected injuries (a full season at risk), over the injury mix:
      H1: expected days beyond each deductible / 365 x w
      H2: P(the injury ends his career) x years of contract left x w.
    """
    excess = lognormal_excess(mu.ravel(), sigma, deductibles)
    out = p[["club", "wy"]].copy()
    w = p["share"] * p["lam"]
    for d in deductibles:
        out[f"e_h1_d{d}_pct"] = w * excess[d].reshape(mu.shape).mean(axis=1) / 365 * 100
    pc = pce.mean(axis=1)
    for y in years_left:
        out[f"e_h2_y{y}_pct"] = w * pc * y * 100
    out["injuries_expected"] = p["lam"]
    out = out.groupby(["club", "wy"]).sum().reset_index()
    return out.merge(p.groupby(["club", "wy"])["matched_share"].first().reset_index(), on=["club", "wy"])


def implied_loading(pct_by_d, premium_pct, h2_pct, deductibles=LOADING_D):
    """If the policy had deductible d, the loading the premium would have to contain: premium / expected payout - 1.
    Negative means the premium is below the expected payout (so that deductible is too short)."""
    return {d: premium_pct / (pct_by_d[d] + h2_pct) - 1 for d in deductibles}


def player_losses(p, mu, pce, sigma, deductibles=FIT_D, years_left=YEARS_LEFT):
    """expected_losses() before summing over the squad: one row per player-season, in % of the club's payroll."""
    excess = lognormal_excess(mu.ravel(), sigma, deductibles)
    w = p["share"] * p["lam"]
    out = p[["club", "wy", "share"]].copy()
    for d in deductibles:
        out[f"h1_d{d}"] = w * excess[d].reshape(mu.shape).mean(axis=1) / 365 * 100
    pc = pce.mean(axis=1)
    for y in years_left:
        out[f"h2_y{y}"] = w * pc * y * 100
    return out


def insured_premium(pl, payroll, threshold, deductible, years_left, loading, top_n=None):
    """Predicted premium (% of payroll) per club-season if only players with a wage of at least `threshold` (EUR m)
    are insured. A player's wage is his payroll share x the club's payroll. Also returns the number insured."""
    m = pl.merge(payroll, on=["club", "wy"])
    insured = (m["share"] * m["payroll"]) >= threshold
    if top_n is not None:
        insured = m.groupby(["club", "wy"])["share"].rank(ascending=False, method="first") <= top_n
    m = m.assign(n=insured.astype(int), cost=np.where(insured, m[f"h1_d{deductible}"] + m[f"h2_y{years_left}"], 0.0))
    g = m.groupby(["club", "wy"]).agg(cost=("cost", "sum"), n_insured=("n", "sum"))
    g["predicted_pct"] = (1 + loading) * g["cost"]
    return g.reset_index()


def fit_threshold(pl, fin, years_left=2, thresholds=THRESHOLDS, deductibles=FIT_D, loadings=FIT_LOADINGS):
    """For each loading, the (wage threshold, deductible) that best reproduces the clubs' average premiums, as the
    mean squared log error across clubs. The threshold-free fit (threshold 0: everyone insured) is shown alongside,
    so you can see whether limiting cover to the best-paid players explains the premiums better."""
    payroll = fin[["club", "wy", "payroll"]]
    actual = fin.groupby("club")["premium_pct"].mean()
    rows = []
    for T in thresholds:
        for d in deductibles:
            for th in loadings:
                g = insured_premium(pl, payroll, T, d, years_left, th)
                club = g.groupby("club")[["predicted_pct", "n_insured"]].mean()
                err = np.mean(np.log(club["predicted_pct"].clip(lower=1e-4) / actual.reindex(club.index)) ** 2)
                rows.append({"loading": th, "threshold_m": T, "deductible": d, "error": err, **{
                    f"pred_{c}": club.loc[c, "predicted_pct"] for c in club.index}, **{
                    f"n_{c}": club.loc[c, "n_insured"] for c in club.index}})
    return pd.DataFrame(rows)


def fit_top_n(pl, fin, years_left=2, top_ns=TOP_N, deductibles=FIT_D, loadings=FIT_LOADINGS):
    """Same search as fit_threshold, but the rule is "the N best-paid players of each club are insured"."""
    payroll = fin[["club", "wy", "payroll"]]
    actual = fin.groupby("club")["premium_pct"].mean()
    rows = []
    for n in top_ns:
        for d in deductibles:
            for th in loadings:
                g = insured_premium(pl, payroll, 0, d, years_left, th, top_n=n)
                club = g.groupby("club")["predicted_pct"].mean()
                err = np.mean(np.log(club.clip(lower=1e-4) / actual.reindex(club.index)) ** 2)
                rows.append({"loading": th, "top_n": n, "deductible": d, "error": err,
                             **{f"pred_{c}": club[c] for c in club.index}})
    return pd.DataFrame(rows)


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


def load_squad(last):
    """Club squads with payroll shares (by market value on 1 July), for the seasons with usable injury records."""
    mv = sdb.query("""SELECT ext_player_id, value_date, market_value FROM market_values
                      WHERE sport = 'Soccer' AND market_value > 0 AND value_date IS NOT NULL""")
    squad = wage_shares(squads(), mv)
    return squad[squad["wy"].between(freq.DEFAULT_START_YEAR, last)]


def fit_squad_inputs(squad):
    """Fit the frequency, severity and career-ending models on all the data and return squad_inputs() plus sigma."""
    sev_data = sev.build_dataset()
    fdata = freq.build_dataset(start_year=freq.DEFAULT_START_YEAR, leagues=freq.TOP5_LEAGUES)
    poisson = freq.PoissonGLM().fit(fdata)
    lognormal = sev.LognormalModel().fit(sev_data)
    career_model = career.LogisticGLM().fit(career.build_dataset())
    p, mu, pce = squad_inputs(squad, fdata, poisson, lognormal, career_model, sev_data)
    return p, mu, pce, lognormal.sigma


def main():
    fin = load_financials()
    last = career.last_complete_season_start(sev.player_seasons("Soccer")).year
    fin = fin[(fin["wy"] >= freq.DEFAULT_START_YEAR) & (fin["wy"] <= last)]
    inj = sev.injury_features("Soccer")
    inj = inj[(inj["days_missed"] > 0) & (inj["days_missed"] <= 1000) & (inj["nature"] != "Not an injury")]
    inj = inj.assign(pid=inj["ext_player_id"])
    squad = load_squad(last)
    pd.set_option("display.width", 220)

    # --- realized ---
    grid_h1 = realized_h1(squad, inj, GRID)
    h1 = realized_h1(squad, inj)
    rate, n_events, n_seasons = career_ending_rate(last - career.FOLLOW_UP_SEASONS)
    print(f"career-ending: {n_events} events in {n_seasons:,} top-5 player-seasons = {rate:.3%} per season")
    t = compare(fin, h1, rate)
    print("\nREALIZED, per club, as a multiple of the premium (1.0 = matches):")
    print(summary(t).T.to_string())

    # --- expected, from the models ---
    print("\nFitting the frequency, severity and career-ending models on all the data ...", flush=True)
    p, mu, pce, sigma = fit_squad_inputs(squad)
    e = expected_losses(p, mu, pce, sigma, deductibles=sorted(set(DEDUCTIBLES + GRID + list(LOADING_D))))
    te = fin.merge(e, on=["club", "wy"]).merge(h1, on=["club", "wy"])
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
        for y in YEARS_LEFT:
            exp_curve = pd.Series({d: gg[f"e_h1_d{d}_pct"].mean() for d in GRID}, name="e")
            real_curve = grid_h1[grid_h1["club"] == club].set_index("wy").loc[gg["wy"]].filter(like="h1_d").mean()
            real_curve.index = [int(k.split("_d")[1].split("_")[0]) for k in real_curve.index]
            h2 = gg[f"e_h2_y{y}_pct"].mean()
            r[f"expected, {y} yr left"] = implied_deductible(exp_curve, r["premium_pct"], h2)
            r[f"realized, {y} yr left"] = implied_deductible(real_curve.reindex(GRID), r["premium_pct"], rate * y * 100)
        rows.append(r)
    print("\nIMPLIED DEDUCTIBLE (days): the waiting period at which H1 + H2 equals the average premium")
    print("(0 = covers every day; inf = even a 365-day deductible leaves the losses above the premium; nan = H2 alone exceeds it)")
    imp = pd.DataFrame(rows).set_index("club")
    print(imp.round({c: 0 for c in imp.columns if c != "premium_pct"}).round({"premium_pct": 2}).to_string())

    # --- the loading the premium would contain at each possible deductible ---
    print("\nLOADING the premium would contain if the policy had this deductible (model-based expected payout, 2 yr left;")
    print("0% = the premium equals the expected payout; negative = the premium is below the expected payout):")
    lrows = []
    for club, gg in te.groupby("club"):
        curve = {d: gg[f"e_h1_d{d}_pct"].mean() for d in LOADING_D}
        lrows.append({"club": club, **implied_loading(curve, gg["premium_pct"].mean(), gg["e_h2_y2_pct"].mean())})
    print((pd.DataFrame(lrows).set_index("club") * 100).round(0).rename(columns=lambda d: f"{d}-day").to_string())

    # --- only the best-paid players insured? ---
    pl = player_losses(p, mu, pce, sigma)
    fits = fit_threshold(pl, fin)
    print("\nONLY THE BEST-PAID PLAYERS INSURED? Best (wage threshold in EUR m a year, deductible in days) per loading,")
    print("chosen so one pair reproduces all four clubs' average premiums. 'no threshold' = everyone insured:")
    for th, g in fits.groupby("loading"):
        best, flat = g.loc[g["error"].idxmin()], g[g["threshold_m"] == 0].loc[lambda x: x["error"].idxmin()]
        print(f"  loading {th:.0%}: threshold {best.threshold_m:g}m, deductible {best.deductible:g} days, error {best.error:.3f}"
              f"   |  no threshold: deductible {flat.deductible:g} days, error {flat.error:.3f}")
    tn = fit_top_n(pl, fin)
    print("\nALTERNATIVE: the N best-paid players of each club are insured. Best (N, deductible) per loading:")
    for th, g in tn.groupby("loading"):
        b = g.loc[g["error"].idxmin()]
        print(f"  loading {th:.0%}: top {b.top_n:g} players, deductible {b.deductible:g} days, error {b.error:.3f}   |   "
              + ", ".join(f"{c[5:]} {b[c]:.2f}" for c in tn.columns if c.startswith("pred_")))
    th0 = fits["loading"].median()
    best = fits[fits["loading"] == th0].loc[lambda x: x["error"].idxmin()]
    clubs = sorted(c[5:] for c in fits.columns if c.startswith("pred_"))
    actual = fin.groupby("club")["premium_pct"].mean()
    print(f"\nAt the best pair for a {th0:.0%} loading (threshold {best.threshold_m:g}m, deductible {best.deductible:g} days):")
    print(pd.DataFrame({"actual premium %": actual, "predicted %": [best[f"pred_{c}"] for c in clubs],
                        "players insured (avg)": [best[f"n_{c}"] for c in clubs]}, index=clubs).round(2).to_string())


if __name__ == "__main__":
    main()
