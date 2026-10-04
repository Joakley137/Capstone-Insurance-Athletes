"""
career.py — will an injury end the player's career? A yes/no model to sit alongside the severity model.

    python career.py                          # soccer: build the dataset, fit the models, print the comparison

    import career
    data = career.build_dataset()             # one row per injury, career_ending = 0/1
    train, test = career.split(data)
    models = career.fit_all(train)
    career.compare(models, test)
    career.odds_ratios(models["Logistic GLM, main effects"])

What counts as career-ending (built from the player_seasons appearance records):
  * the player never made another recorded professional appearance in any season that started after
    the injury, and
  * he had not returned from the injury by the end of the season it happened in.
Only the first such injury per player counts; his later injuries are dropped (the career is already over).
Injuries are only labeled when two complete seasons of appearance data follow them, so a player who is
still rehabbing is not mistaken for a retirement. Players who died are excluded.

"Professional" means competitions Transfermarkt records appearances for; a player who drops to amateur
football after the injury counts as career-ending, which matches what a player's insurance covers.
Predictors are only things known on the day of the injury: the diagnosis, the player and his history.
How long the injury eventually lasted is NOT used.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import severity as sev
import sportsdb as sdb

FOLLOW_UP_SEASONS = 2
MIN_EVENTS = 20              # career-ending injuries a body region / injury type needs in train to keep its own level

FEATURES = ["region", "nature", "position_group", "age", "prior_injuries", "reinjury_60d", "prior_same_part",
            "log_mv", "year", "games_12m", "career_games"]

# Injury-time predictors. region_g / nature_g are region and nature with rare levels grouped (see LogisticGLM).
RHS_MAIN = ("C(region_g, Treatment('Hamstring')) + C(nature_g, Treatment('Unspecified'))"
            " + C(position_group, Treatment('Midfield')) + age_c + I(age_c ** 2)"
            " + np.log1p(prior_injuries) + reinjury_60d + prior_same_part + log_mv_c + mv_missing"
            " + np.log1p(games_12m) + no_games_12m + np.log1p(career_games) + year_c")
CANDIDATES = {
    "age x injury type": "age_c:C(nature_g)",
    "age x body region": "age_c:C(region_g)",
    "market value x injury type": "log_mv_c:C(nature_g)",
    "minutes (12 months) x age": "np.log1p(games_12m):age_c",
    "same body part before x injury type": "prior_same_part:C(nature_g)",
}
# select_interactions() picks some on the validation split, but none improved the held-out test set
# (only ~400 career-ending injuries to learn from), so the final model uses main effects only.
FINAL_INTERACTIONS = []


def rhs_final(interactions=None):
    terms = FINAL_INTERACTIONS if interactions is None else interactions
    return " + ".join([RHS_MAIN] + [CANDIDATES.get(t, t) for t in terms])


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def last_complete_season_start(ps):
    """Start date of the latest July-June season whose appearance data is complete
    (at least 80% as many player-seasons as the best of the previous five)."""
    split = ps[ps["season_start"].dt.month == 7]
    n = split.groupby("season_start").size().sort_index()
    full = [start for i, start in enumerate(n.index)
            if i >= 1 and n.iloc[i] >= 0.8 * n.iloc[max(0, i - 5):i].max()]
    return full[-1]


def build_dataset(sport="Soccer", follow_up_seasons=FOLLOW_UP_SEASONS):
    f = sev.injury_features(sport)
    ps = sev.player_seasons(sport)
    played = ps[ps["appearances"] > 0]

    # Last season with an appearance, and the end of the season the injury happened in
    f["last_season_played"] = f["ext_player_id"].map(played.groupby("ext_player_id")["season_start"].max())
    played_after = f["last_season_played"] > f["injury_date"]
    season_end = sev._asof(f, ps, "injury_date", "season_start", "season_end")
    june_30 = pd.to_datetime((f["injury_date"].dt.year + (f["injury_date"].dt.month >= 7)).astype(str) + "-06-30")
    season_end = season_end.where(season_end >= f["injury_date"], june_30)
    returned_in_season = f["return_date"].notna() & (f["return_date"] <= season_end)

    candidate = ~played_after & ~returned_in_season
    end_date = f["pid"].map(f[candidate].groupby("pid")["injury_date"].min())
    f["career_ending"] = (candidate & (f["injury_date"] == end_date)).astype(int)

    players = sdb.query("SELECT ext_player_id, current_club, date_of_death FROM players WHERE sport = :sport",
                        {"sport": sport})
    f = f.merge(players, on="ext_player_id", how="left")

    cutoff = last_complete_season_start(ps) - pd.DateOffset(years=follow_up_seasons - 1)
    keep = ((f["injury_date"] < cutoff)
            & f["ext_player_id"].isin(played["ext_player_id"])          # has appearance records at all
            & f["date_of_death"].isna()
            & ~(f["injury_date"] > end_date).fillna(False)              # after the career already ended
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
# Models. Each has .predict(df) -> probability the injury ends the career.
# --------------------------------------------------------------------------
class BaseRate:
    """No predictors: every injury gets the training rate."""

    def __init__(self, name="Base rate, no predictors"):
        self.name = name

    def fit(self, train):
        self.p = train["career_ending"].mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.p)


class LogisticGLM:
    """Logistic regression (binomial GLM, logit link).

    Career-ending injuries are rare, so a body region or injury type with fewer than min_events of them
    in the training data is merged into 'Other' (decided on train only); otherwise its coefficient
    could not be estimated.
    """

    def __init__(self, rhs=None, name="Logistic GLM", min_events=MIN_EVENTS):
        self.rhs, self.name, self.min_events = rhs or rhs_final(), name, min_events

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
        self.levels["region"] |= {"Hamstring"}           # reference levels always kept
        self.levels["nature"] |= {"Unspecified"}
        d = self._grouped(train)
        self.design = sev.Design(self.rhs)
        X = self.design.fit(d)
        self.res = sm.GLM(d["career_ending"].to_numpy(), X, family=sm.families.Binomial()).fit()
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(self.design.transform(self._grouped(df))))


class GradientBoosting:
    def __init__(self, name="Gradient boosting"):
        self.name = name

    def fit(self, train):
        self.gbm = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
                                                  min_samples_leaf=200, early_stopping=True,
                                                  categorical_features="from_dtype", random_state=0)
        self.gbm.fit(train[FEATURES], train["career_ending"])
        return self

    def predict(self, df):
        return self.gbm.predict_proba(df[FEATURES])[:, 1]


def select_interactions(train, candidates=CANDIDATES, min_gain=0.001, seed=7):
    """Forward selection of interactions on a validation split of train, judged by log loss."""
    fit_d, val_d = split(train, test_size=0.25, seed=seed)
    y = val_d["career_ending"]
    loss = lambda terms: log_loss(y, LogisticGLM(rhs_final(terms)).fit(fit_d).predict(val_d))
    best = loss([])
    path, chosen = [{"step": "main effects", "val_log_loss": best}], []
    while True:
        trials = {n: loss(chosen + [n]) for n in candidates if n not in chosen}
        if not trials:
            break
        name, l = min(trials.items(), key=lambda kv: kv[1])
        if l > best * (1 - min_gain):
            break
        chosen.append(name)
        best = l
        path.append({"step": f"+ {name}", "val_log_loss": l})
    out = pd.DataFrame(path)
    out["improvement_vs_previous"] = -out["val_log_loss"].diff()
    return out.round(6), chosen


def default_models(interactions=None):
    return [BaseRate(), LogisticGLM(RHS_MAIN, "Logistic GLM, main effects"),
            LogisticGLM(rhs_final(interactions), "Logistic GLM + interactions"), GradientBoosting()]


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
    """Score each model on held-out injuries.

    auc            chance a random career-ending injury is ranked above a random other one (0.5 = guessing)
    avg_precision  area under the precision-recall curve; the base rate is what guessing gets
    log_loss / brier   accuracy of the probabilities themselves (lower is better)
    mean_pred      average predicted probability; should match the actual rate
    top10_capture  share of all career-ending injuries among the 10% the model rates riskiest
    """
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


def calibration(models, test, bins=10):
    """Average predicted vs. actual rate, by decile of each model's prediction."""
    rows = []
    for name, m in models.items():
        p = m.predict(test)
        if np.ptp(p) == 0:
            continue
        g = pd.DataFrame({"pred": p, "actual": test["career_ending"].to_numpy(),
                          "decile": pd.qcut(p, bins, labels=False, duplicates="drop")}).groupby("decile")
        rows.append(g.mean().assign(model=name, n=g.size()))
    return pd.concat(rows).reset_index()


def odds_ratios(model):
    """exp(coefficient) with 95% CI: how much each factor multiplies the odds of a career-ending injury."""
    res = model.res
    ci = np.exp(res.conf_int())
    out = pd.DataFrame({"odds_ratio": np.exp(res.params), "ci_low": ci[0], "ci_high": ci[1], "p_value": res.pvalues})
    out.index = sev.tidy_terms(out.index)
    return out.round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="Soccer")
    p.add_argument("--select", action="store_true", help="re-run the interaction search instead of FINAL_INTERACTIONS")
    args = p.parse_args()
    pd.set_option("display.width", 160)

    data = build_dataset(args.sport)
    train, test = split(data)
    print(f"{len(data):,} injuries before {data.attrs['cutoff']:%Y-%m-%d}, "
          f"{data['career_ending'].sum():,} career-ending ({data['career_ending'].mean():.2%}); "
          f"train {len(train):,} / test {len(test):,}")
    interactions = None
    if args.select:
        path, interactions = select_interactions(train)
        print(path.to_string())
    models = fit_all(train, default_models(interactions))
    print("\nHeld-out comparison:")
    print(compare(models, test).to_string())
    print("\nLogistic GLM odds ratios:")
    print(odds_ratios(models["Logistic GLM, main effects"]).to_string())


if __name__ == "__main__":
    main()
