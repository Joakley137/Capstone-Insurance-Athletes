"""
career.py — logistic GLM: does an injury end the player's career?

    python -m model.career [--sport Soccer]

    from model import career
    train, test = career.split(career.build_dataset())
    models = career.fit_all(train)
    career.compare(models, test)

Career-ending = no professional appearance in any later season and no return within the injury's
season (first such injury per player; needs two complete follow-up seasons; deceased players excluded).
Predictors are known on the injury date only. The GLM is checked against a no-predictor base rate on held-out players.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from model import severity as sev
from database import sportsdb as sdb

FOLLOW_UP_SEASONS = 2
MIN_EVENTS = 20  # min career-ending events in train for a region/type to keep its own level

# region_g / nature_g: rare levels grouped into 'Other' (see LogisticGLM)
RHS_MAIN = ("C(region_g, Treatment('Hamstring')) + C(nature_g, Treatment('Unspecified'))"
            " + C(position_group, Treatment('Midfield')) + age_c + I(age_c ** 2)"
            " + np.log1p(prior_injuries) + reinjury_60d + prior_same_part"
            " + np.log1p(games_12m) + no_games_12m + np.log1p(career_games) + year_c")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def last_complete_season_start(ps):
    """Latest July season with >= 80% of the player-seasons of the best of the previous five."""
    split = ps[ps["season_start"].dt.month == 7]
    n = split.groupby("season_start").size().sort_index()
    full = [start for i, start in enumerate(n.index)
            if i >= 1 and n.iloc[i] >= 0.8 * n.iloc[max(0, i - 5):i].max()]
    return full[-1]


def build_dataset(sport="Soccer", follow_up_seasons=FOLLOW_UP_SEASONS):
    f = sev.injury_features(sport)
    ps = sev.player_seasons(sport)
    played = ps[ps["appearances"] > 0]

    f["last_season_played"] = f["ext_player_id"].map(played.groupby("ext_player_id")["season_start"].max())
    played_after = f["last_season_played"] > f["injury_date"]
    season_end = sev._asof(f, ps, "injury_date", "season_start", "season_end")
    june_30 = pd.to_datetime((f["injury_date"].dt.year + (f["injury_date"].dt.month >= 7)).astype(str) + "-06-30")
    season_end = season_end.where(season_end >= f["injury_date"], june_30)
    returned_in_season = f["return_date"].notna() & (f["return_date"] <= season_end)

    candidate = ~played_after & ~returned_in_season
    end_date = f["pid"].map(f[candidate].groupby("pid")["injury_date"].min())
    f["career_ending"] = (candidate & (f["injury_date"] == end_date)).astype(int)

    players = sdb.query("SELECT ext_player_id, date_of_death FROM players")
    f = f.merge(players, on="ext_player_id", how="left")

    cutoff = last_complete_season_start(ps) - pd.DateOffset(years=follow_up_seasons - 1)
    keep = ((f["injury_date"] < cutoff)
            & f["ext_player_id"].isin(played["ext_player_id"])
            & f["date_of_death"].isna()
            & ~(f["injury_date"] > end_date).fillna(False)  # career already over
            & (f["nature"] != "Not an injury")
            & f["age"].between(15, 45))
    df = f[keep].reset_index(drop=True)
    df.attrs["cutoff"] = cutoff
    for c in ["region", "nature", "position_group"]:
        df[c] = df[c].astype("category")
    return df


def split(df, test_size=0.2, seed=42):
    return sev.split(df, test_size, seed)


# --------------------------------------------------------------------------
# Models: .fit(train) / .predict(df) -> P(career-ending)
# --------------------------------------------------------------------------
class BaseRate:
    """Training-set rate for every injury."""

    def __init__(self, name="Base rate, no predictors"):
        self.name = name

    def fit(self, train):
        self.p = train["career_ending"].mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.p)


class LogisticGLM:
    """Binomial GLM, logit link. Regions/types with < min_events events in train are pooled as 'Other'."""

    def __init__(self, rhs=None, name="Logistic GLM", min_events=MIN_EVENTS):
        self.rhs, self.name, self.min_events = rhs or RHS_MAIN, name, min_events

    def _grouped(self, df):
        out = df.copy()
        for c, keep in self.levels.items():
            v = out[c].astype(str)
            out[f"{c}_g"] = pd.Categorical(v.where(v.isin(keep), "Other"), categories=sorted(keep | {"Other"}))
        return out

    def fit(self, train):
        events = train[train["career_ending"] == 1]
        self.levels = {c: set(events[c].astype(str).value_counts().loc[lambda s: s >= self.min_events].index)
                       for c in ["region", "nature"]}
        self.levels["region"] |= {"Hamstring"}  # reference levels
        self.levels["nature"] |= {"Unspecified"}
        d = self._grouped(train)
        self.design = sev.Design(self.rhs)
        X = self.design.fit(d)

        # ==================================================================
        # GLM FIT — logistic (binomial, logit link)
        # ==================================================================
        self.res = sm.GLM(d["career_ending"].to_numpy(), X, family=sm.families.Binomial()).fit()
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(self.design.transform(self._grouped(df))))


def default_models():
    return [BaseRate(), LogisticGLM(RHS_MAIN, "Logistic GLM, main effects")]


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
    """Held-out AUC, average precision, log loss, Brier, mean prediction, top-decile capture."""
    y = test["career_ending"].to_numpy()
    rows = []
    for name, m in models.items():
        p = np.clip(m.predict(test), 1e-6, 1 - 1e-6)
        top = p >= np.quantile(p, 0.9) if np.ptp(p) > 0 else np.zeros(len(p), bool)
        rows.append({"model": name,
                     "auc": roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5,
                     "avg_precision": average_precision_score(y, p),
                     "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
                     "mean_pred": p.mean(), "top10_capture": y[top].sum() / y.sum()})
    out = pd.DataFrame(rows).set_index("model")
    out.loc["(actual)", "mean_pred"] = y.mean()
    return out.round(5)


coefficients = sev.coefficients


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="Soccer")
    args = p.parse_args()
    pd.set_option("display.width", 160)

    data = build_dataset(args.sport)
    train, test = split(data)
    print(f"{len(data):,} injuries before {data.attrs['cutoff']:%Y-%m-%d}, "
          f"{data['career_ending'].sum():,} career-ending ({data['career_ending'].mean():.2%}); "
          f"train {len(train):,} / test {len(test):,}")
    models = fit_all(train)
    print("\nHeld-out comparison:")
    print(compare(models, test).to_string())
    print("\nLogistic GLM coefficients:")
    print(coefficients(models["Logistic GLM, main effects"]).to_string())


if __name__ == "__main__":
    main()
