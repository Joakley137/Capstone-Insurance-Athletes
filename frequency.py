"""
frequency.py — model how often a player gets injured: injuries per day at risk, from the player's profile and history.

    python frequency.py                       # soccer, top-5 leagues: build the dataset, fit every model, print the comparison
    python frequency.py --all-leagues         # every league in the data

    import frequency as freq
    data = freq.build_dataset(leagues=freq.TOP5_LEAGUES)   # one row per player-season ("window"), with an injury count and exposure
    train, test = freq.split(data)            # 80/20, split by player so no player is in both
    models = freq.fit_all(train)
    freq.compare(models, test)
    freq.relativities(models["Negative binomial GLM"])     # exp(coef) multipliers on the injury rate
    freq.recurrence_table(data)               # observed rate by number of earlier injuries

Frequency x severity: the expected number of injuries in a window times the expected days per injury
(severity.py) is the expected days lost, and 1 - exp(-expected injuries) is the chance of at least one.
Which injury it will be (severity classes) is decided by severity.py / predict.py once an injury happens.

One row per player and July-June season, from the player's first to last season with an appearance, so
seasons with no injuries (and seasons missed entirely) are in the table. Without them the baseline
rate can't be estimated. Predictors are only things known on 1 July, before the season starts. Minutes
played in the season itself are NOT used (an injury cuts them short).

Exposure is days at risk: the 365 days of the season minus days already out injured (at least 30).
Needs injuries, players, player_seasons and market_values loaded (python build_db.py soccer data/football-datasets).
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance, roc_auc_score

import career
import severity as sev
import sportsdb as sdb

TOP5_LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]     # Premier League, LaLiga, Serie A, Bundesliga, Ligue 1 (source codes)
DEFAULT_START_YEAR = 2012    # injury records are thin before this: about 2% of players have one recorded in a season up to 2006, 10-15% by 2012
MIN_EXPOSURE = 30            # days at risk; a player who is injured all season still counts as a month at risk
MAJOR_DAYS = 90              # an injury this long counts as "major" in the history
LOG_MV_CENTER = sev.LOG_MV_CENTER

RHS_MAIN = ("np.log1p(prior_injuries) + np.log1p(prior_last_year) + np.log1p(prior_days_out_1y) + prior_major"
            " + carry_in + pos_attack + pos_defender + pos_goalkeeper + age_c + I(age_c ** 2)"
            " + log_mv_c + mv_missing + np.log1p(games_prior) + no_games_prior + np.log1p(career_games) + year_c")
BOOST_FEATURES = ["prior_injuries", "prior_last_year", "prior_days_out_1y", "prior_major", "carry_in",
                  "pos_attack", "pos_defender", "pos_goalkeeper", "age", "log_mv_c", "mv_missing",
                  "games_prior", "career_games", "year"]


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def season_year(dates):
    """July-June season a date falls in, named by the year it starts."""
    d = pd.DatetimeIndex(dates)
    return pd.Series(np.where(d.month >= 7, d.year, d.year - 1), index=getattr(dates, "index", None))


def _july(year):
    return pd.to_datetime(pd.DataFrame({"year": np.asarray(year, dtype=int), "month": 7, "day": 1}))


def make_windows(inj, ps, players, mv, last_year, start_year=None, in_scope=None):
    """The modeling table from plain frames (so it can be tested without a database).

    inj:     injuries with ext_player_id, injury_date, days_missed (open ones already set to days out so far)
    ps:      player_seasons: ext_player_id, season_start, season_end, minutes, appearances
    players: ext_player_id, date_of_birth, position_group
    mv:      market_values: ext_player_id, value_date, market_value
    last_year: last season (by start year) with complete appearance data; later seasons are dropped
    in_scope: appearances (same columns as ps) that decide who is in the table and from when to when, e.g. only
              top-5-league appearances. A player is kept from his first to his last season with such an
              appearance, including seasons in between spent injured or in another league. Minutes
              and history still count every competition. Default: any appearance.
    """
    # Minutes per player and July-June season (calendar-year leagues go to the season holding their midpoint)
    def with_wy(d):
        d = d.copy()
        d["wy"] = season_year(d["season_start"] + (d["season_end"] - d["season_start"]) / 2).to_numpy()
        return d
    ps = with_wy(ps)
    minutes = ps.groupby(["ext_player_id", "wy"])["minutes"].sum()
    played = (ps if in_scope is None else with_wy(in_scope))
    played = played[played["appearances"] > 0]
    span = played.groupby("ext_player_id")["wy"].agg(["min", "max"])
    span["max"] = span["max"].clip(upper=last_year)
    span = span[span["min"] <= span["max"]]

    rows = [(pid, y) for pid, a, b in zip(span.index, span["min"], span["max"]) for y in range(a, b + 1)]
    w = pd.DataFrame(rows, columns=["pid", "wy"])
    w["minutes"] = pd.MultiIndex.from_frame(w).map(minutes).fillna(0).to_numpy() if len(w) else []
    w["t0"] = _july(w["wy"]).astype("datetime64[ns]")

    # Injuries: count per window, days out per window (an injury can run over into later windows)
    inj = inj.assign(pid=inj["ext_player_id"], wy=season_year(inj["injury_date"]).to_numpy())
    inj = inj[inj["pid"].isin(span.index)].copy()
    start = inj["injury_date"]
    end = start + pd.to_timedelta(inj["days_missed"], unit="D")
    days_out, carry = [], []
    for k in range(5):                                     # injuries are capped at 1,000 days, so <= 4 later windows
        ws = _july(inj["wy"] + k)
        we = _july(inj["wy"] + k + 1)
        ov = (np.minimum(end, we) - np.maximum(start, ws)).dt.days.clip(lower=0)
        days_out.append(pd.DataFrame({"pid": inj["pid"], "wy": inj["wy"] + k, "days_out": ov,
                                      "carry_in": (ov > 0) & (k > 0)}))
    days_out = pd.concat(days_out)
    days_out = days_out.groupby(["pid", "wy"]).agg(days_out=("days_out", "sum"), carry_in=("carry_in", "max"))
    inj["major"] = inj["days_missed"] >= MAJOR_DAYS
    by_win = inj.groupby(["pid", "wy"]).agg(injuries=("pid", "size"), n_major=("major", "sum"))
    pre = inj[inj["wy"] < inj["pid"].map(span["min"])].groupby("pid").agg(n=("pid", "size"), m=("major", "sum"))

    w = w.join(by_win, on=["pid", "wy"]).join(days_out, on=["pid", "wy"])
    w[["injuries", "n_major", "days_out"]] = w[["injuries", "n_major", "days_out"]].fillna(0)
    w["carry_in"] = w["carry_in"].eq(True).astype(int)
    w = w.sort_values(["pid", "wy"]).reset_index(drop=True)
    w["days_out"] = w["days_out"].clip(upper=365)
    w["exposure_days"] = (365 - w["days_out"]).clip(lower=MIN_EXPOSURE)

    # History as of 1 July: everything strictly before this window
    g = w.groupby("pid")
    w["prior_injuries"] = g["injuries"].cumsum() - w["injuries"] + w["pid"].map(pre["n"]).fillna(0)
    w["prior_major"] = ((g["n_major"].cumsum() - w["n_major"] + w["pid"].map(pre["m"]).fillna(0)) > 0).astype(int)
    w["prior_last_year"] = g["injuries"].shift().fillna(0)
    w["prior_days_out_1y"] = g["days_out"].shift().fillna(0)
    w["games_prior"] = g["minutes"].shift().fillna(0) / 90
    w["no_games_prior"] = (w["games_prior"] <= 0).astype(int)
    w["career_games"] = (g["minutes"].cumsum() - w["minutes"]) / 90

    # Player: age and position on 1 July, market value in the 2 years before
    w = w.merge(players[["ext_player_id", "date_of_birth", "position_group"]]
                .drop_duplicates("ext_player_id"), left_on="pid", right_on="ext_player_id", how="left")
    w["age"] = (w["t0"] - pd.to_datetime(w["date_of_birth"])).dt.days / 365.25
    w["position_group"] = w["position_group"].fillna("Unknown")
    w["log_mv"] = np.nan
    if not mv.empty:
        mv = mv.assign(value_date=pd.to_datetime(mv["value_date"]).astype("datetime64[ns]")).rename(columns={"ext_player_id": "pid"})
        w = w.sort_values("t0")
        hit = pd.merge_asof(w[["t0", "pid"]].reset_index(), mv.sort_values("value_date"), left_on="t0",
                            right_on="value_date", by="pid", direction="backward",
                            tolerance=pd.to_timedelta(730, unit="D")).set_index("index")
        w["log_mv"] = np.log(hit["market_value"])
        w = w.sort_values(["pid", "wy"])
    w["year"] = w["wy"]
    w["age_c"] = w["age"] - 26
    w["year_c"] = w["year"] - 2015
    w["mv_missing"] = w["log_mv"].isna().astype(int)
    w["log_mv_c"] = (w["log_mv"] - LOG_MV_CENTER).fillna(0)
    for pos in ["Attack", "Defender", "Goalkeeper"]:
        w[f"pos_{pos.lower()}"] = (w["position_group"] == pos).astype(int)

    keep = w["age"].between(15, 45) & (w["wy"] >= (start_year or -np.inf))
    w = w[keep].drop(columns=["ext_player_id", "date_of_birth", "t0"]).reset_index(drop=True)
    w["position_group"] = w["position_group"].astype("category")
    return w


def build_dataset(sport="Soccer", start_year=None, max_days=1000, leagues=None):
    """One row per player-season. Injuries are filtered like the severity table (a real injury,
    at most max_days), so frequency x severity describe the same events.

    start_year: first season to keep, if the injury records start later than the appearance records
    (see coverage()).
    leagues: competition codes (e.g. TOP5_LEAGUES) to restrict to; only players who appear in them are kept
    (see make_windows). Needs competition_id in player_seasons: reload with `python build_db.py soccer-seasons`.
    """
    f = sev.injury_features(sport)
    f = f[(f["days_missed"] > 0) & (f["days_missed"] <= max_days) & (f["nature"] != "Not an injury")]
    ps = sev.player_seasons(sport)
    players = sdb.query("SELECT ext_player_id, date_of_birth, position_group FROM players WHERE sport = :sport",
                        {"sport": sport})
    mv = sdb.query("""SELECT ext_player_id, value_date, market_value FROM market_values
                      WHERE sport = :sport AND market_value > 0 AND value_date IS NOT NULL""", {"sport": sport})
    in_scope = None
    if leagues:
        marks = ",".join(f":l{i}" for i in range(len(leagues)))
        in_scope = sdb.query(f"""SELECT ext_player_id, season_start, season_end, appearances FROM player_seasons
                                 WHERE sport = :sport AND competition_id IN ({marks}) AND season_end IS NOT NULL""",
                             {"sport": sport, **{f"l{i}": l for i, l in enumerate(leagues)}})
        for c in ["season_start", "season_end"]:
            in_scope[c] = pd.to_datetime(in_scope[c])
        in_scope["minutes"] = 0.0
    return make_windows(f, ps, players, mv, career.last_complete_season_start(ps).year, start_year, in_scope)


def split(df, test_size=0.2, seed=42):
    """Train/test split by player, so all of a player's seasons are on one side."""
    return sev.split(df, test_size, seed)


def coverage(df):
    """Injuries per 365 days at risk, by season. A season where this drops sharply is one where the injury
    records are thin; start the dataset after it with build_dataset(start_year=...)."""
    g = df.groupby("year").agg(players=("pid", "size"), injuries=("injuries", "sum"), exposure=("exposure_days", "sum"))
    g["injuries_per_player_year"] = (g["injuries"] / g["exposure"] * 365).round(3)
    g["share_with_injury"] = df.groupby("year")["injuries"].apply(lambda s: (s > 0).mean()).round(3)
    return g


def recurrence_table(df):
    """The simple version of the question: observed injury rate by number of earlier injuries.
    Not adjusted for age, minutes or anything else; the GLM relativities are."""
    bins = pd.cut(df["prior_injuries"], [-1, 0, 1, 2, 4, 8, np.inf], labels=["0", "1", "2", "3-4", "5-8", "9+"])
    g = df.groupby(bins, observed=True).agg(player_seasons=("pid", "size"), injuries=("injuries", "sum"),
                                            exposure=("exposure_days", "sum"),
                                            share_injured=("injuries", lambda s: (s > 0).mean()))
    g["injuries_per_365_days"] = g["injuries"] / g["exposure"] * 365
    return g.drop(columns="exposure").round(3)


# --------------------------------------------------------------------------
# Models. Each has .predict(df) -> expected injuries over df's exposure, .rate(df) -> expected injuries
# per 365 days at risk; count models also .pmf(df) for the likelihood.
# --------------------------------------------------------------------------
class _Counts:
    def rate(self, df):
        return self.predict(df.assign(exposure_days=365.0))

    def p_injury(self, df):
        """Chance of at least one injury over df's exposure."""
        return 1 - self.pmf(df, 0)


class BaseRate(_Counts):
    """No predictors: every player-day has the training rate."""

    def __init__(self, name="Base rate, no predictors"):
        self.name = name

    def fit(self, train):
        self.lam = train["injuries"].sum() / train["exposure_days"].sum()
        return self

    def predict(self, df):
        return self.lam * df["exposure_days"].to_numpy()

    def pmf(self, df, k):
        return stats.poisson.pmf(k, self.predict(df))


class PoissonGLM(_Counts):
    """Poisson regression, log link, log(days at risk) as offset: exp(coef) multiplies the injury rate."""

    def __init__(self, rhs=RHS_MAIN, name="Poisson GLM"):
        self.rhs, self.name = rhs, name

    def fit(self, train):
        self.design = sev.Design(self.rhs)
        X = self.design.fit(train)
        self.res = sm.GLM(train["injuries"].to_numpy(), X, family=sm.families.Poisson(),
                          offset=np.log(train["exposure_days"].to_numpy())).fit()
        self.beta = self.res.params.to_numpy()
        return self

    def predict(self, df):
        X = self.design.transform(df)
        return np.exp(X.to_numpy() @ self.beta + np.log(df["exposure_days"].to_numpy()))

    def pmf(self, df, k):
        return stats.poisson.pmf(k, self.predict(df))


class NegBinGLM(PoissonGLM):
    """Negative binomial (variance mu + alpha mu^2): for injuries that cluster in some players more than the
    predictors explain. Same relativities interpretation as the Poisson."""

    def __init__(self, rhs=RHS_MAIN, name="Negative binomial GLM"):
        super().__init__(rhs, name)

    def fit(self, train):
        self.design = sev.Design(self.rhs)
        X = self.design.fit(train)
        start = PoissonGLM.fit(PoissonGLM(self.rhs), train).res.params.to_numpy()
        res = sm.NegativeBinomial(train["injuries"].to_numpy(), X, offset=np.log(train["exposure_days"].to_numpy())
                                  ).fit(start_params=np.append(start, 0.5), disp=0, maxiter=200)
        self.res, self.alpha = res, float(res.params.iloc[-1])
        self.beta = res.params.to_numpy()[:-1]
        return self

    def pmf(self, df, k):
        n = 1 / self.alpha
        return stats.nbinom.pmf(k, n, n / (n + self.predict(df)))


class BoostedPoisson(_Counts):
    """Gradient boosting with Poisson loss on injuries per day at risk (weighted by days at risk): an
    accuracy benchmark with no formula. No likelihood, so it is scored on deviance only."""

    def __init__(self, name="Gradient boosting (Poisson loss)"):
        self.name = name

    def fit(self, train):
        self.m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=200, random_state=0)
        e = train["exposure_days"].to_numpy()
        self.m.fit(train[BOOST_FEATURES], train["injuries"].to_numpy() / e, sample_weight=e)
        return self

    def predict(self, df):
        return self.m.predict(df[BOOST_FEATURES]) * df["exposure_days"].to_numpy()

    def pmf(self, df, k):
        return stats.poisson.pmf(k, self.predict(df))


def default_models():
    return [BaseRate(), PoissonGLM(), NegBinGLM(), BoostedPoisson()]


def fit_all(train, models=None):
    out = {}
    for m in models or default_models():
        print(f"fitting {m.name} ...", flush=True)
        out[m.name] = m.fit(train)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def compare(models, test):
    """Held-out scores per model.

    poisson_dev: mean Poisson deviance per player-season (lower is better)
    log_lik:     average log-likelihood of the observed count (higher is better; not for boosting)
    auc_any:     how well the injury rate ranks player-seasons with an injury above those without (0.5 = no better than chance)
    mean_pred:   average expected injuries; compare with the (actual) row
    """
    y = test["injuries"].to_numpy()
    rows = [{"model": "(actual)", "mean_pred": y.mean()}]
    for name, m in models.items():
        mu = m.predict(test)
        ll = np.log(np.clip(m.pmf(test, y), 1e-300, None)).mean() if not isinstance(m, BoostedPoisson) else np.nan
        rows.append({"model": name, "poisson_dev": mean_poisson_deviance(y, np.clip(mu, 1e-9, None)), "log_lik": ll,
                     "auc_any": roc_auc_score(y > 0, m.rate(test)), "mean_pred": mu.mean()})
    return pd.DataFrame(rows).set_index("model").round(4)


def calibration(models, test, bins=10):
    """Average predicted vs. actual injuries by decile of predicted injuries (test set)."""
    out = []
    for name, m in models.items():
        mu = m.predict(test)
        d = pd.DataFrame({"pred": mu, "actual": test["injuries"].to_numpy()})
        d["bin"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False)
        g = d.groupby("bin")[["pred", "actual"]].mean().reset_index(drop=True)
        out.append(g.assign(model=name))
    return pd.concat(out, ignore_index=True)


def relativities(model):
    """exp(coefficient): the multiplier on the injury rate for a one-unit change, with a 95% interval."""
    keep = [c for c in model.res.params.index if c != "alpha"]
    ci = model.res.conf_int().loc[keep]
    out = pd.DataFrame({"multiplier": np.exp(model.res.params[keep]), "ci_low": np.exp(ci.iloc[:, 0]),
                        "ci_high": np.exp(ci.iloc[:, 1]), "p_value": model.res.pvalues[keep]})
    return out.round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                   help="first season to use; injury records are thin before about 2012 (see coverage())")
    p.add_argument("--all-leagues", action="store_true", help="keep players from every league, not just the top 5")
    args = p.parse_args()
    data = build_dataset(start_year=args.start_year, leagues=None if args.all_leagues else TOP5_LEAGUES)
    train, test = split(data)
    print(f"{len(data):,} player-seasons, {data.pid.nunique():,} players, {int(data.injuries.sum()):,} injuries")
    print(recurrence_table(data), "\n")
    models = fit_all(train)
    print(compare(models, test).to_string())


if __name__ == "__main__":
    main()
