"""
pricing.py — per-season probabilities for insuring a player: injury, injury beyond the deferment, career-ending.

    python -m model.pricing                          # example quotes and calibration checks

    from model import pricing
    pricing.quote(27, position="Defender", league="Premier League", minutes_last_season=2500,
                  career_minutes=15000, prior_injuries=4)

Method — three fitted models, one set of player inputs:
  lambda = expected injuries this season           negative-binomial GLM (frequency.py), dispersion alpha
  q(D)   = sum_k w_k P(days > D | cell k, player, league)   tiered gamma GLM + its per-tier spread (severity.py)
  c      = sum_k w_k P(career-ending | cell k, player)   logistic GLM (career.py)
  A future injury's type is unknown, so the per-injury models are averaged over the injury mix:
  w_k = share of injuries in each body region x injury type cell (which fixes both the severity rating class
  and the career model's region/type). combine() turns lambda, q, c into season probabilities with the NB
  zero probability.

Population: what is priced is a big-five player-season, so the severity GLM and the injury mix are fit on
the insured injuries only (insured_injuries): the injuries the frequency model counts, i.e. real injuries,
0 < days <= 1000, ages 15-45, starting in a big-five player-season from 2015/16 (league = the player's
league that July-June season). These are shorter than injuries in all leagues (13% vs 21% over 60 days):
fitting q on every league over-predicted P(injury > 60 days) by 1.44x in the back-test. The severity GLM
here also has a league term: leagues record injuries differently (the Premier League records fewer but
longer injuries, Serie A / Bundesliga more and shorter ones), so league moves lambda and q in opposite
directions; without it the Premier League's covered rate was under-predicted by a third and Serie A's
over-predicted. The career GLM stays on injuries in all leagues: the insured population has only 25
career-ending injuries, too few for its region / type levels.

Inputs (quote): age, position, league, minutes_last_season, career_minutes, prior_injuries,
days_out_last_season, date. "Last season" is treated as "the last 12 months": minutes_last_season feeds
both the frequency model (minutes last season) and the career model (minutes in the last 12 months).
prior_injuries feeds all three models; league feeds lambda and q; days_out_last_season only lambda;
career_minutes and date only c. Missing inputs take typical values (frequency model's training medians / most
common position and league; career minutes from the insured injuries) and are listed under `assumed`.

Fitting: fit(sev_train, career_train, freq_train) for back-tests; models() is the all-data version
(predict.models(), cached in model/fitted/injury_models.pkl).
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from model import career
from model import frequency as freq
from model import predict
from model import severity as sev

DEFERMENT_DAYS = 60
SENSITIVITY_DAYS = (30, 60, 90, 180)
SEVERITY_LEAGUE = " + C(league, Treatment('GB1'))"     # league term added to the severity GLM
LEAGUE_NAMES = {"premier league": "GB1", "laliga": "ES1", "la liga": "ES1", "serie a": "IT1",
                "bundesliga": "L1", "ligue 1": "FR1"}


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def injury_mix(data):
    """w_k: share of injuries per (region, nature) cell -> DataFrame region, nature, w."""
    w = data.groupby(["region", "nature"], observed=True).size()
    return (w / w.sum()).rename("w").reset_index().astype({"region": str, "nature": str})


def insured_injuries(sev_data, seasons):
    """Injuries (rows of severity.build_dataset()) that start in one of `seasons` (frequency.build_dataset()
    player-seasons: pid, season, league), with that season's league added. Season = July-June."""
    season = sev_data["injury_date"].dt.year - (sev_data["injury_date"].dt.month < 7)
    league = seasons.drop_duplicates(["pid", "season"]).set_index(["pid", "season"])["league"].astype(str)
    out = sev_data.assign(league=pd.MultiIndex.from_arrays([sev_data["pid"], season]).map(league.get))
    out = out[out["league"].notna()].reset_index(drop=True)
    out["league"] = pd.Categorical(out["league"], categories=freq.LEAGUES)
    return out


def fit(sev_train, career_train, freq_train):
    """Fit on training data (e.g. train players only): the negative-binomial frequency GLM on freq_train
    (frequency.build_dataset()), the tiered gamma GLM (with league) and injury mix on the sev_train injuries
    in those player-seasons (insured_injuries; sev_train from severity.build_dataset()), and the logistic
    career GLM on career_train (career.build_dataset(), all leagues).
    Returns the bundle used by injury_probs / quote (same keys as predict.models())."""
    sev_train = insured_injuries(sev_train, freq_train)
    cells = sev_train.groupby(["region", "nature"], observed=True)["injury_class"].first().astype(str)
    b = {
        "population": "big five",
        "tiered": sev.TieredGammaGLM(extra=SEVERITY_LEAGUE).fit(sev_train),
        "career": career.LogisticGLM(career.RHS_MAIN).fit(career_train),
        "class_of": {(r, n): c for (r, n), c in cells.items()},
        "classes": sorted(sev_train["injury_class"].astype(str).unique()),
        "typical": {c: float(sev_train[c].median()) for c in ["prior_injuries", "games_12m", "career_games"]},
        "career_last_year": int(career_train["year"].max()),
        "mix": injury_mix(sev_train),
        "n_insured_injuries": len(sev_train),
        "frequency": freq.CountGLM(freq.RHS_MAIN, freq.FINAL, "nb").fit(freq_train),
    }
    return b


def models(refit=False):
    """All-data bundle (fitted and cached by predict.py)."""
    return predict.models(refit)


# --------------------------------------------------------------------------
# Per-injury probabilities, averaged over the mix
# --------------------------------------------------------------------------
def _tail(tiered, df, days):
    """P(days > D) and E[days | days > D] per row, from the tier's actual/predicted quantiles
    (same spread as TieredGammaGLM.range_sf, vectorised)."""
    mu = tiered.predict(df)
    tiers = df["injury_class"].astype(str).map(tiered.tier_of).to_numpy()
    sf, cond = np.empty(len(df)), np.empty(len(df))
    for t in np.unique(tiers):
        i = tiers == t
        r = tiered.ratio_quantiles[t]
        above = r[None, :] > (days / mu[i])[:, None]                  # grid points above D
        n = above.sum(1)
        sf[i] = n / len(r)
        cond[i] = np.where(n > 0, mu[i] * (above * r).sum(1) / np.maximum(n, 1), days)
    return sf, cond


def _cell_rows(b, player):
    """One feature row per mix cell for this player (player inputs as predict.injury, incl. league)."""
    base, assumed = predict._row(b, body_part="Hamstring", injury_type="Strain / muscle", **player)
    mix = b["mix"]
    df = pd.DataFrame([base] * len(mix))
    df["region"], df["nature"] = mix["region"].to_numpy(), mix["nature"].to_numpy()
    df["injury_class"] = [predict._rating_class(b, r, n) for r, n in zip(df["region"], df["nature"])]
    return df, mix["w"].to_numpy(), assumed


def _career_p(b, df):
    """As predict._predict_frame: don't extrapolate the year trend past the career data."""
    cdf = df.assign(year_c=np.minimum(df["year"], b["career_last_year"]) - 2015)
    return b["career"].predict(cdf)


def injury_probs(b=None, deferments=SENSITIVITY_DAYS, **player):
    """Per-injury probabilities for a random future injury of this player -> Series:
    q_D = P(days > D), mean_days_over_D = E[days | days > D], career_ending = c, expected_days."""
    b = b or models()
    df, w, assumed = _cell_rows(b, player)
    out = {"expected_days": w @ b["tiered"].predict(df)}
    for d in deferments:
        sf, cond = _tail(b["tiered"], df, d)
        out[f"q_{d}"] = w @ sf
        out[f"mean_days_over_{d}"] = (w * sf) @ cond / max(w @ sf, 1e-12)
    out["career_ending"] = w @ _career_p(b, df)
    s = pd.Series(out)
    s.attrs["assumed"] = assumed
    return s


# --------------------------------------------------------------------------
# Season probabilities
# --------------------------------------------------------------------------
def p_any(mean, alpha=None):
    """P(N >= 1) for a count with this mean: negative binomial (variance mean + alpha mean^2)
    1 - (1 + alpha mean)^(-1/alpha), as frequency.p_no_injury; alpha None or 0 -> Poisson 1 - e^-mean."""
    mean = np.asarray(mean, float)
    return -np.expm1(-mean) if not alpha else 1 - (1 + alpha * mean) ** (-1 / alpha)


def combine(lam, q, c, alpha=None):
    """Season probabilities from lambda (expected injuries per player-season), q = P(injury > deferment),
    c = P(career-ending | injury) and the frequency model's NB dispersion alpha (None = Poisson).

    Thinning: if each injury independently exceeds the deferment with probability q, covered injuries
    have mean lam q and (NB) the same alpha, so P(at least one) = 1 - (1 + alpha lam q)^(-1/alpha);
    likewise career-ending ones with lam c. Assumes injury count, type and severity are independent
    (the type mix doesn't change with how often the player is hurt). A career-ending injury is counted
    regardless of later injuries, so lam c slightly overstates (c is small; negligible)."""
    return pd.Series({
        "expected_injuries": lam,
        "expected_covered_injuries": lam * q,
        "p_any_injury": float(p_any(lam, alpha)),
        "p_covered_injury": float(p_any(lam * q, alpha)),
        "p_career_ending": float(p_any(lam * c, alpha)),
    })


def league_code(league):
    """Big-five code (GB1, ES1, IT1, L1, FR1) from a code or name like 'Premier League'."""
    code = LEAGUE_NAMES.get(str(league).strip().lower(), str(league).strip().upper())
    if code not in freq.LEAGUES:
        raise ValueError(f"league must be one of {freq.LEAGUES} or {sorted(LEAGUE_NAMES)}")
    return code


def quote(age, position=None, league=None, minutes_last_season=None, career_minutes=None, prior_injuries=None,
          days_out_last_season=None, date=None, deferment=DEFERMENT_DAYS, b=None):
    """Season quote for one player -> Series: lambda, q, c and the combined season probabilities.
    See the module doc for which input feeds which model; `assumed` lists the filled-in inputs."""
    b = b or models()
    nb = b["frequency"]
    t = nb.typical
    given = {"position": position, "league": league, "minutes_last_season": minutes_last_season,
             "career_minutes": career_minutes, "prior_injuries": prior_injuries,
             "days_out_last_season": days_out_last_season}
    typical = {**t, "career_minutes": b["typical"]["career_games"] * 90}
    v = {k: typical[k] if x is None else x for k, x in given.items()}
    assumed = [f"{k.replace('_', ' ')} = {v[k]:,.0f}" if isinstance(v[k], float) else f"{k} = {v[k]}"
               for k, x in given.items() if x is None]
    v["position"], v["league"] = str(v["position"]).strip().title(), league_code(v["league"])
    when = date or dt.date.today().isoformat()

    row = freq.player_row(age, v["position"], v["league"], v["minutes_last_season"], v["prior_injuries"],
                          v["days_out_last_season"], typical=t)
    lam = float(freq.expected_injuries(nb, row)[0])
    p = injury_probs(b, sorted({deferment, *SENSITIVITY_DAYS}), age=age, position=v["position"], league=v["league"],
                     minutes_last_12_months=v["minutes_last_season"], career_minutes=v["career_minutes"],
                     prior_injuries=v["prior_injuries"], date=when)
    q, c = p[f"q_{deferment}"], p["career_ending"]
    out = pd.concat([pd.Series({"deferment_days": deferment, "alpha": nb.alpha, "q": q,
                                "mean_days_if_covered": p[f"mean_days_over_{deferment}"], "c": c}),
                     combine(lam, q, c, nb.alpha),
                     p[[f"q_{d}" for d in SENSITIVITY_DAYS]]])
    out["assumed"] = "; ".join(assumed) or "nothing"
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def calibration(b=None, n=5000, seed=0):
    """Mix-averaged q(D) and c over a sample of injured players vs the empirical rates (q on the insured
    injuries, c on all injuries as the career model). Each sampled injury contributes its player's inputs;
    its own type is replaced by the mix. In-sample (all-data fit): runs/backtest.py is the held-out check."""
    b = b or models()
    data = insured_injuries(sev.build_dataset(), freq.build_dataset())
    cdata = career.build_dataset()
    mix = b["mix"]
    tiered = b["tiered"]

    def mixed(players, fn):
        rows = players.loc[players.index.repeat(len(mix))].reset_index(drop=True)
        rows["region"] = np.tile(mix["region"].to_numpy(), len(players))
        rows["nature"] = np.tile(mix["nature"].to_numpy(), len(players))
        rows["injury_class"] = [predict._rating_class(b, r, k) for r, k in zip(rows["region"], rows["nature"])]
        for c in ["region", "nature", "position_group"]:
            rows[c] = rows[c].astype(str)
        return np.average(fn(rows).reshape(len(players), len(mix)) @ mix["w"].to_numpy())

    done = sev.completed(data)
    ps = done.sample(n, random_state=seed)
    out = {d: {"model": mixed(ps, lambda r: _tail(tiered, r, d)[0]),
               "empirical": (done["days_missed"] > d).mean()} for d in SENSITIVITY_DAYS}
    cs = cdata.sample(n, random_state=seed)
    out["career_ending"] = {"model": mixed(cs, lambda r: _career_p(b, r)), "empirical": cdata["career_ending"].mean()}
    return pd.DataFrame(out).T


EXAMPLES = {
    "27yo PL defender, regular starter": dict(age=27, position="Defender", league="Premier League",
                                              minutes_last_season=2500, career_minutes=15000, prior_injuries=4),
    "33yo defender": dict(age=33, position="Defender", minutes_last_season=2200, career_minutes=45000,
                          prior_injuries=10),
    "22yo Bundesliga attacker": dict(age=22, position="Attack", league="Bundesliga", minutes_last_season=1500,
                                     career_minutes=5000, prior_injuries=1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deferment", type=int, default=DEFERMENT_DAYS)
    p.add_argument("--no-check", action="store_true", help="skip the calibration check")
    a = p.parse_args()
    pd.set_option("display.width", 160)
    quotes = pd.DataFrame({k: quote(deferment=a.deferment, **v) for k, v in EXAMPLES.items()})
    print(quotes.drop("assumed").astype(float).round(4).to_string())
    print("\nFilled in:\n" + quotes.loc["assumed"].to_string())
    if not a.no_check:
        print("\nMix-averaged model vs empirical rate:")
        print(calibration().round(4).to_string())


if __name__ == "__main__":
    main()
