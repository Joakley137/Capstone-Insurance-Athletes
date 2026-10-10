"""
frequency.py — injury frequency: expected number of injuries a player has in a season (count GLM).

    python -m model.frequency                       # fit, print held-out comparison, calibration and coefficients

    from model import frequency as freq
    train, test = freq.split(freq.build_dataset())
    models = freq.fit_all(train)
    freq.compare(models, test)
    nb = models[freq.FINAL]
    freq.expected_injuries(nb, freq.player_row(27, position="Defender", league="GB1", minutes_last_season=2500))

Exposure unit: one player-season (July-June) in a big-five first division (Premier League, LaLiga,
Serie A, Bundesliga, Ligue 1 = player_seasons.league), seasons 2015/16 to the last complete season,
age 15-44 at season start. Every player listed in a big-five squad that season counts, including
those with 0 minutes, so the population is not selected on playing (an injured-all-season player stays in).

Why this population: Transfermarkt injury records are dense only in the top leagues. Over all
competitions 43% of player-seasons belong to players with no injury ever recorded, and the recorded
rate per season ranges from 1.5 (Bundesliga) to 0.05 (League Two) to 0.03 (National League): the
gap is recording, not risk. In the big five, 88-99% of players have a recorded injury and the
rate per season is stable from 2015/16 (about 0.9-1.1); earlier seasons are thinner (0.04 before 2007).
Bias: rates are for big-five players and will overstate risk elsewhere only if recording there were as
complete (it is not); differences between the five leagues (Bundesliga about 2x Ligue 1) are partly
recording practice, so the league term is a "rate as recorded" adjustment; injuries Transfermarkt
misses for big-five players (minor knocks especially) are missing here too. Prior-injury counts use the
same records, so they are undercounted for older players whose early careers predate dense recording.

n_injuries counts the same injuries the severity model and pricing.py use: real injuries
(sev.nature != "Not an injury"), 0 < days_missed <= 1000, player age 15-45 at injury, starting
between 1 July and 30 June of the season. Injuries still open at the data date count as
events (days so far is used for the <= 1000 rule); seasons used are complete, so these are few.

Predictors are known at the start of the season: league, position, age (centered at 26, + square),
log(1 + full games last season) and a no-games-last-season flag (minutes in any competition in seasons
ending in the 12 months before the season starts), log(1 + prior injuries), log(1 + days out to
injuries that started last season). The season's own minutes are NOT used: they are unknown at quote
time and partly a consequence of injury (players who play 2700+ minutes have half the rate of those
on 900-2700). Every unit is one season, so there is no exposure offset; part-season exposure
(transfers in/out of the big five mid-season) is a limitation.

Counts are overdispersed (variance 1.7 vs mean 1.0; Poisson GLM Pearson chi2/df 1.3), so the final model is a negative
binomial (NB2) GLM, log link, alpha by MLE. With NB, P(N = 0) = (1 + alpha mu)^(-1/alpha), and thinning
by a per-injury probability q gives NB(mean mu q, same alpha): use p_no_injury(model, df, q).
"""
import argparse
import functools
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupShuffleSplit

from model import career
from model import severity as sev
from database import sportsdb as sdb

FIRST_SEASON = 2015          # 2015/16
MAX_DAYS = 1000              # same injury population as severity.build_dataset
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]
POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]

RHS_MAIN = ("C(league, Treatment('GB1')) + C(position_group, Treatment('Midfield')) + age_c + I(age_c ** 2)"
            " + np.log1p(games_last_season) + no_games_last_season + np.log1p(prior_injuries)"
            " + np.log1p(days_out_last_season)")
FINAL = "Negative binomial GLM"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def covered_injuries(sport="Soccer"):
    """Injury events in the severity / pricing population (open injuries included, days so far)."""
    f = sev.injury_features(sport)
    keep = ((f["days_missed"] > 0) & (f["days_missed"] <= MAX_DAYS) & (f["nature"] != "Not an injury")
            & f["age"].between(15, 45))
    inj = f.loc[keep, ["pid", "injury_date", "days_missed", "censored"]].reset_index(drop=True)
    inj["season"] = inj["injury_date"].dt.year - (inj["injury_date"].dt.month < 7)   # July-June season
    return inj


def build_dataset(sport="Soccer", first_season=FIRST_SEASON, last_season=None):
    """One row per big-five player-season with n_injuries and start-of-season predictors (see module doc)."""
    ps = sev.player_seasons(sport)
    league = sdb.query("SELECT ext_player_id, season_start, league FROM player_seasons WHERE league IS NOT NULL")
    league["season_start"] = pd.to_datetime(league["season_start"])
    last_season = last_season or career.last_complete_season_start(ps).year
    df = league[league["season_start"].dt.year.between(first_season, last_season)].copy()
    df["pid"] = df["ext_player_id"]
    df["season"] = df["season_start"].dt.year

    players = sdb.query("SELECT ext_player_id, date_of_birth, position_group FROM players")
    df = df.merge(players, on="ext_player_id", how="left")
    df["age"] = (df["season_start"] - pd.to_datetime(df["date_of_birth"])).dt.days / 365.25
    df = df[df["age"].between(15, 44)].copy()
    df["position_group"] = df["position_group"].fillna("Unknown")

    # Outcome and injury history
    inj = covered_injuries(sport)
    by_season = inj.groupby(["pid", "season"])
    df = df.join(by_season.size().rename("n_injuries"), on=["pid", "season"])
    prev = by_season["days_missed"].sum().rename("days_out_last_season").reset_index()
    prev["season"] += 1
    df = df.merge(prev, on=["pid", "season"], how="left")
    inj = inj.sort_values("injury_date")
    inj["n_before"] = inj.groupby("pid").cumcount() + 1
    df = df.sort_values("season_start")
    hist = pd.merge_asof(df[["pid", "season_start"]], inj[["pid", "injury_date", "n_before"]],
                         left_on="season_start", right_on="injury_date", by="pid",
                         direction="backward", allow_exact_matches=False)
    df["prior_injuries"] = hist["n_before"].to_numpy()

    # Minutes in seasons that ended in the 12 months before this season (July season or calendar year)
    end = ps["season_end"]
    ps = ps.assign(season=np.where(end.dt.month == 6, end.dt.year, end.dt.year + 1))
    mins = ps.groupby(["ext_player_id", "season"])["minutes"].sum().rename("minutes_last_season")
    df = df.join(mins, on=["ext_player_id", "season"])

    fill = ["n_injuries", "days_out_last_season", "prior_injuries", "minutes_last_season"]
    df[fill] = df[fill].fillna(0)
    df["n_injuries"] = df["n_injuries"].astype(int)
    return add_features(df.sort_values(["pid", "season"]).reset_index(drop=True))


def add_features(df):
    """Derived model columns (also used by player_row)."""
    df["age_c"] = df["age"] - 26
    df["games_last_season"] = df["minutes_last_season"] / 90          # full-game equivalents
    df["no_games_last_season"] = (df["minutes_last_season"] <= 0).astype(int)
    df["league"] = pd.Categorical(df["league"], categories=LEAGUES)
    df["position_group"] = pd.Categorical(df["position_group"], categories=POSITIONS + ["Unknown"])
    return df


@functools.lru_cache(maxsize=2)
def _severity_players(test_size, seed):
    sd = sev.build_dataset()
    _, te = sev.split(sd, test_size, seed)
    return frozenset(sd["pid"]), frozenset(te["pid"])


def split(df, test_size=0.2, seed=42):
    """Train/test split grouped by player. A player in the severity data keeps his severity split
    (held out there = held out here); players without severity injuries are split the same way."""
    sev_all, sev_test = _severity_players(test_size, seed)
    test = df["pid"].isin(sev_test).to_numpy()
    other = np.flatnonzero(~df["pid"].isin(sev_all).to_numpy())
    if len(other):
        _, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                     .split(other, groups=df["pid"].to_numpy()[other]))
        test[other[te]] = True
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)


# --------------------------------------------------------------------------
# Models: .fit(train) / .predict(df) -> expected injuries; .logpmf, .p_zero
# --------------------------------------------------------------------------
class CountGLM:
    """Poisson or negative binomial (NB2) GLM, log link. NB alpha by MLE (sm.NegativeBinomial)."""

    def __init__(self, rhs=RHS_MAIN, name="Poisson GLM", family="poisson"):
        self.rhs, self.name, self.family = rhs, name, family

    def fit(self, train):
        self.design = sev.Design(self.rhs)
        X = self.design.fit(train)
        y = train["n_injuries"].to_numpy()

        # ==================================================================
        # GLM FIT — Poisson or negative binomial, log link
        # ==================================================================
        if self.family == "nb":
            with np.errstate(all="ignore"):                     # log(0) during the line search
                self.alpha = float(sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200).params["alpha"])
            fam = sm.families.NegativeBinomial(alpha=self.alpha)
        else:
            self.alpha = 0.0
            fam = sm.families.Poisson()
        self.res = sm.GLM(y, X, family=fam).fit()
        self.dispersion = float(self.res.pearson_chi2 / self.res.df_resid)
        self.typical = typical_values(train)
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(self.design.transform(df)))

    def logpmf(self, df):
        y, mu = df["n_injuries"].to_numpy(), self.predict(df)
        if self.alpha == 0:
            return stats.poisson.logpmf(y, mu)
        n = 1 / self.alpha
        return stats.nbinom.logpmf(y, n, n / (n + mu))

    def p_zero(self, df, q=1.0):
        """P(no injury), or P(no covered injury) when each injury is covered with probability q."""
        mu = self.predict(df) * q
        return np.exp(-mu) if self.alpha == 0 else (1 + self.alpha * mu) ** (-1 / self.alpha)


def default_models():
    return [CountGLM("1", "Mean rate, no predictors", "nb"), CountGLM(RHS_MAIN, "Poisson GLM", "poisson"),
            CountGLM(RHS_MAIN, FINAL, "nb")]


def fit_all(train, models=None):
    out = {}
    for m in models or default_models():
        print(f"  fitting {m.name} ...", flush=True)
        out[m.name] = m.fit(train)
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def compare(models, test):
    """Held-out mean log-likelihood (higher better), Poisson deviance, MAE, mean prediction, P(0 injuries)."""
    y = test["n_injuries"].to_numpy()
    rows = []
    for name, m in models.items():
        mu = m.predict(test)
        rows.append({"model": name, "log_lik": np.mean(m.logpmf(test)), "poisson_dev": mean_poisson_deviance(y, mu),
                     "mae": np.mean(np.abs(y - mu)), "mean_pred": mu.mean(), "p_zero": m.p_zero(test).mean()})
    out = pd.DataFrame(rows).set_index("model")
    out.loc["(actual)", ["mean_pred", "p_zero"]] = [y.mean(), np.mean(y == 0)]
    return out.round(4)


def calibration(model, test, bins=10):
    """By predicted decile: mean predicted vs actual injuries, and predicted vs actual P(0 injuries)."""
    mu = model.predict(test)
    d = pd.DataFrame({"decile": pd.qcut(pd.Series(mu).rank(method="first"), bins, labels=range(1, bins + 1)),
                      "pred": mu, "actual": test["n_injuries"].to_numpy(), "p0_pred": model.p_zero(test),
                      "p0_actual": (test["n_injuries"].to_numpy() == 0)})
    return d.groupby("decile", observed=True).agg(n=("pred", "size"), pred=("pred", "mean"), actual=("actual", "mean"),
                                                  p0_pred=("p0_pred", "mean"), p0_actual=("p0_actual", "mean")).round(3)


def coefficients(model):
    """GLM coefficients (log scale), rate ratios exp(coef) and p-values."""
    out = sev.coefficients(model)
    out.insert(1, "rate_ratio", np.exp(out["coef"]).round(4))
    return out


# --------------------------------------------------------------------------
# For pricing
# --------------------------------------------------------------------------
def typical_values(df):
    return {"age": float(df["age"].median()), "minutes_last_season": float(df["minutes_last_season"].median()),
            "prior_injuries": float(df["prior_injuries"].median()),
            "days_out_last_season": float(df["days_out_last_season"].median()),
            "position": df["position_group"].astype(str).mode()[0], "league": df["league"].astype(str).mode()[0]}


def player_row(age=None, position=None, league=None, minutes_last_season=None, prior_injuries=None,
               days_out_last_season=None, typical=None):
    """One-row frame for predict/p_zero. Omitted inputs take `typical` (model.typical: training medians,
    most common position/league) and are listed in .attrs['assumed']. league: GB1, ES1, IT1, L1 or FR1."""
    typical = typical or typical_values(build_dataset())
    given = {"age": age, "position": position, "league": league, "minutes_last_season": minutes_last_season,
             "prior_injuries": prior_injuries, "days_out_last_season": days_out_last_season}
    assumed = [k for k, v in given.items() if v is None]
    v = {k: typical[k] if given[k] is None else given[k] for k in given}
    if v["league"] not in LEAGUES:
        raise ValueError(f"league must be one of {LEAGUES}")
    row = add_features(pd.DataFrame({"age": [float(v["age"])], "position_group": [v["position"]], "league": [v["league"]],
                                     "minutes_last_season": [float(v["minutes_last_season"])],
                                     "prior_injuries": [float(v["prior_injuries"])],
                                     "days_out_last_season": [float(v["days_out_last_season"])]}))
    row.attrs["assumed"] = assumed
    return row


def expected_injuries(model, df):
    """λ: expected injuries in the season."""
    return model.predict(df)


def p_no_injury(model, df, q=1.0):
    """P(no injury with per-injury probability q in the season): NB or Poisson as fitted."""
    return model.p_zero(df, q)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="Soccer")
    args = p.parse_args()
    pd.set_option("display.width", 160)
    t0 = time.time()

    data = build_dataset(args.sport)
    train, test = split(data)
    print(f"{len(data):,} big-five player-seasons {data['season'].min()}/{data['season'].min() + 1 - 2000}"
          f"-{data['season'].max()}/{data['season'].max() + 1 - 2000}, {data['pid'].nunique():,} players, "
          f"{data['n_injuries'].sum():,} injuries ({data['n_injuries'].mean():.3f} per season, "
          f"variance {data['n_injuries'].var():.3f}); train {len(train):,} / test {len(test):,}")
    models = fit_all(train)
    nb = models[FINAL]
    print(f"\nPoisson GLM Pearson chi2/df = {models['Poisson GLM'].dispersion:.2f}; NB alpha = {nb.alpha:.3f}")
    print("\nHeld-out comparison (test players):")
    print(compare(models, test).to_string())
    print(f"\n{FINAL} calibration by predicted decile (test):")
    print(calibration(nb, test).to_string())
    print(f"\n{FINAL} coefficients:")
    print(coefficients(nb).to_string())
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
