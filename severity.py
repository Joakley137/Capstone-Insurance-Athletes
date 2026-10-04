"""
severity.py — model injury severity (days missed) and compare a gamma GLM against alternatives.

    python severity.py                        # soccer: build the dataset, fit every model, print the comparison
    python severity.py --select               # also re-run the interaction search (a few minutes)

    import severity as sev
    data = sev.build_dataset()                # one row per injury, with predictors and a rating class
    train, test = sev.split(data)             # 80/20, split by player so no player is in both
    models = sev.fit_all(train)
    sev.compare(models, test)
    sev.relativities(models["Gamma GLM, rating classes + interactions"])   # exp(coef) multipliers

    path, chosen = sev.select_interactions(train)   # forward search on a validation split of train

Needs injuries, players, player_seasons and market_values loaded (python build_db.py soccer data/football-datasets).
The career-ending model in career.py builds on injury_features() here.
"""
import argparse
import functools
import gc
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import WeibullAFTFitter
from patsy import build_design_matrices, dmatrix
from scipy import optimize, special, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_gamma_deviance
from sklearn.model_selection import GroupShuffleSplit

import sportsdb as sdb

# What kind of injury, from the description; first match wins. Illness is left to body_part
# (body_part = 'Illness' already marks it), so the two categoricals don't duplicate each other.
NATURE = [
    (r"\b(fitness|rest\b|suspen)", "Not an injury"),
    (r"\b(ruptur|tear|torn)", "Tear / rupture"),
    (r"\b(fractur|broken|break|crack)", "Fracture"),
    (r"\b(surgery|operation)", "Surgery"),
    (r"\b(ligament|capsul|cartilage|meniscus|disloc|sprain|syndesmo)", "Ligament / joint"),
    (r"\b(tendon|tendin|inflamm|irritat|bursitis|pubalgia)", "Tendon / inflammation"),
    (r"\b(strain|pull|muscle|muscular|fatigue|cramp)", "Strain / muscle"),
    (r"\b(knock|bruise|contusion|dead leg)", "Knock / bruise"),
    (r"\b(concussion)", "Concussion"),
]

# Body parts too rare to price on their own, merged with their anatomical neighbours
REGIONS = {"Quadriceps": "Thigh", "Back": "Back / neck", "Neck": "Back / neck", "Hand": "Arm / hand",
           "Wrist": "Arm / hand", "Elbow": "Arm / hand", "Chest": "Trunk", "Ribs": "Trunk", "Core": "Trunk"}

MIN_CLASS_N = 300            # injuries needed for a body region x injury type cell to be its own rating class
REF_CLASS = "Hamstring (unspecified)"
LOG_MV_CENTER = np.log(500_000)

FEATURES = ["body_part", "nature", "position_group", "age", "prior_injuries", "reinjury_60d", "prior_same_part",
            "log_mv", "year", "offseason", "games_12m", "career_games"]
CATEGORICAL = ["body_part", "region", "nature", "injury_class", "position_group"]

# Everything about the player and the circumstances; the injury itself is added by RHS_MAIN / RHS_CLASS
# Position enters as 0/1 columns (Midfield = all zero) so that position x body region adds only
# interaction terms; a categorical x categorical term would also add body-region columns that
# overlap the rating classes and blur their relativities.
RHS_PLAYER = ("pos_attack + pos_defender + pos_goalkeeper + age_c + I(age_c ** 2)"
              " + np.log1p(prior_injuries) + reinjury_60d + prior_same_part + log_mv_c + mv_missing"
              " + np.log1p(games_12m) + no_games_12m + np.log1p(career_games) + year_c + offseason")
RHS_MAIN = ("C(body_part, Treatment('Hamstring')) + C(nature, Treatment('Unspecified')) + " + RHS_PLAYER)
RHS_CLASS = f"C(injury_class, Treatment('{REF_CLASS}')) + " + RHS_PLAYER

# Interactions tried by select_interactions(), as patsy terms. Each pairs a numeric column that is
# already a main effect with region or injury type, so it adds adjustments relative to hamstring /
# unspecified only.
_R, _N = "C(region, Treatment('Hamstring'))", "C(nature, Treatment('Unspecified'))"
CANDIDATES = {
    "age x body region": f"age_c:{_R}",
    "age x injury type": f"age_c:{_N}",
    "position x body region": f"pos_attack:{_R} + pos_defender:{_R} + pos_goalkeeper:{_R}",
    "offseason x injury type": f"offseason:{_N}",
    "re-injury within 60 days x injury type": f"reinjury_60d:{_N}",
    "same body part before x body region": f"prior_same_part:{_R}",
    "market value x injury type": f"log_mv_c:{_N}",
    "market value x age": "log_mv_c:age_c",
    "minutes (12 months) x age": "np.log1p(games_12m):age_c",
    "minutes (12 months) x body region": f"np.log1p(games_12m):{_R}",
    "year x injury type": f"year_c:{_N}",
    "year x body region": f"year_c:{_R}",
}
# What select_interactions() chose on the soccer data (see severity_models.ipynb); used by default
FINAL_INTERACTIONS = ["year x body region", "market value x injury type", "minutes (12 months) x body region"]


def rhs_final(interactions=None):
    terms = FINAL_INTERACTIONS if interactions is None else interactions
    return " + ".join([RHS_CLASS] + [CANDIDATES.get(t, t) for t in terms])


def nature(text):
    if not isinstance(text, str):
        return "Unspecified"
    t = text.lower()
    for pattern, label in NATURE:
        if re.search(pattern, t):
            return label
    return "Unspecified"


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def player_seasons(sport="Soccer"):
    """Minutes and appearances per player and season (all competitions added together)."""
    ps = sdb.query("""SELECT ext_player_id, season_start, season_end,
                             SUM(minutes) AS minutes, SUM(appearances) AS appearances
                      FROM player_seasons WHERE sport = :sport AND season_end IS NOT NULL
                      GROUP BY ext_player_id, season_start, season_end""", {"sport": sport})
    for c in ["season_start", "season_end"]:
        ps[c] = pd.to_datetime(ps[c])
    return ps


def _asof(left, right, on_left, on_right, value, by="ext_player_id"):
    """For each row of left, `value` from the latest right row (same player) with on_right <= on_left."""
    l = left[["_row", by, on_left]].dropna().sort_values(on_left)
    r = right[[by, on_right, value]].sort_values(on_right)
    hit = pd.merge_asof(l, r, left_on=on_left, right_on=on_right, by=by, direction="backward")
    return left["_row"].map(hit.set_index("_row")[value])


def injury_features(sport="Soccer", as_of=None):
    """Every injury event with its predictors. No rows are dropped and no outcome is filtered,
    so the severity and career-ending datasets can each apply their own rules.

    `censored` marks injuries still open on the data date (no return date, or a projected
    return after it); for those, days_missed is set to days out so far.
    """
    inj = sdb.query("""SELECT injury_id, ext_player_id, player_key, injury_date, return_date, days_missed,
                              body_part, injury_desc, age_at_injury AS age, position_group
                       FROM v_injuries_with_age
                       WHERE sport = :sport AND record_type = 'event'""", {"sport": sport})
    if inj.empty or inj["days_missed"].notna().sum() == 0:
        raise ValueError(f"No {sport} injury events with days_missed are loaded.")

    df = inj.assign(pid=inj["ext_player_id"].fillna(inj["player_key"]),
                    injury_date=pd.to_datetime(inj["injury_date"], errors="coerce"),
                    return_date=pd.to_datetime(inj["return_date"], errors="coerce"))
    df = df.dropna(subset=["pid", "injury_date"]).sort_values(["pid", "injury_date"]).reset_index(drop=True)
    df["_row"] = df.index
    as_of = pd.Timestamp(as_of) if as_of else df["injury_date"].max() + pd.to_timedelta(1, unit="D")
    df.attrs["as_of"] = as_of

    df["censored"] = df["return_date"].isna() | (df["return_date"] > as_of)
    df.loc[df["censored"], "days_missed"] = (as_of - df["injury_date"]).dt.days
    df["days_missed"] = pd.to_numeric(df["days_missed"], errors="coerce")

    # Injury history, using only injuries that started before this one
    g = df.groupby("pid")
    df["prior_injuries"] = g.cumcount()
    since_last = df["injury_date"] - g["return_date"].shift()
    df["reinjury_60d"] = (since_last <= pd.to_timedelta(60, unit="D")).astype(int)     # back < 60 days, or still out
    df["prior_same_part"] = (df["body_part"].notna()
                             & (df.groupby(["pid", "body_part"]).cumcount() > 0)).astype(int)

    # Market value at the time of injury (latest valuation in the 2 years before): player level / club tier
    mv = sdb.query("""SELECT ext_player_id, value_date, market_value FROM market_values
                      WHERE sport = :sport AND market_value > 0 AND value_date IS NOT NULL""", {"sport": sport})
    df["log_mv"] = np.nan
    if not mv.empty:
        mv["value_date"] = pd.to_datetime(mv["value_date"])
        left = df[["_row", "ext_player_id", "injury_date"]].dropna().sort_values("injury_date")
        hit = pd.merge_asof(left, mv.sort_values("value_date"), left_on="injury_date", right_on="value_date",
                            by="ext_player_id", direction="backward", tolerance=pd.to_timedelta(730, unit="D"))
        df["log_mv"] = df["_row"].map(np.log(hit.set_index("_row")["market_value"]))

    # Playing time before the injury. Minutes are per season, so only seasons that had ENDED by the
    # injury date count: a season still in progress would include games played after the injury.
    ps = player_seasons(sport)
    df["games_12m"] = df["career_games"] = 0.0
    if not ps.empty:
        ps = ps.sort_values(["ext_player_id", "season_end"])
        ps["cum_minutes"] = ps.groupby("ext_player_id")["minutes"].cumsum()
        df["_minus_12m"] = df["injury_date"] - pd.to_timedelta(365, unit="D")
        career_now = _asof(df, ps, "injury_date", "season_end", "cum_minutes").fillna(0)
        career_12m_ago = _asof(df, ps, "_minus_12m", "season_end", "cum_minutes").fillna(0)
        df["career_games"] = career_now / 90                   # in full-game equivalents
        df["games_12m"] = (career_now - career_12m_ago) / 90
        df = df.drop(columns="_minus_12m")

    df["body_part"] = df["body_part"].fillna("Unknown")
    df["region"] = df["body_part"].replace(REGIONS)
    df["nature"] = df["injury_desc"].map(nature)
    df["position_group"] = df["position_group"].fillna("Unknown")
    df["year"] = df["injury_date"].dt.year
    df["offseason"] = df["injury_date"].dt.month.isin([6, 7]).astype(int)   # time out includes the summer break

    # Centered versions for the formula models, so the intercept means a typical player
    df["age_c"] = df["age"] - 26
    df["year_c"] = df["year"] - 2015
    df["mv_missing"] = df["log_mv"].isna().astype(int)
    df["log_mv_c"] = (df["log_mv"] - LOG_MV_CENTER).fillna(0)
    df["no_games_12m"] = (df["games_12m"] <= 0).astype(int)
    for pos in ["Attack", "Defender", "Goalkeeper"]:
        df[f"pos_{pos.lower()}"] = (df["position_group"] == pos).astype(int)
    return df


def injury_classes(df, min_n=MIN_CLASS_N):
    """Rating classes: body region x injury type, with rare combinations grouped.

    A combination with at least min_n injuries is its own class ("Knee - Tear / rupture").
    Rarer ones are pooled by injury type across body regions ("Fracture - other sites"), then by
    body region ("Arm / hand - other types"), then into "Other (rare)". Only counts are used, never
    the outcome, so building the classes on the full dataset does not leak test information.
    """
    def label(region, nature_):
        return f"{region} (unspecified)" if nature_ == "Unspecified" else f"{region} - {nature_}"

    cell = pd.Series([label(r, n) for r, n in zip(df["region"], df["nature"])], index=df.index)
    rare = cell.map(cell.value_counts()) < min_n
    by_type = df["nature"].where(rare)
    type_ok = rare & by_type.map(by_type.value_counts()).ge(min_n)
    left = rare & ~type_ok
    by_region = df["region"].where(left)
    region_ok = left & by_region.map(by_region.value_counts()).ge(min_n)

    out = cell.copy()
    out[type_ok] = df.loc[type_ok, "nature"].replace({"Unspecified": "Unspecified injury"}) + " - other sites"
    out[region_ok] = df.loc[region_ok, "region"] + " - other types"
    out[left & ~region_ok] = "Other (rare)"
    return out


def build_dataset(sport="Soccer", max_days=1000, min_class_n=MIN_CLASS_N, as_of=None):
    """The severity modeling table: one row per injury.

    Drops injuries longer than max_days (career-ending or data errors; see career.py), entries that
    are not injuries (rest, fitness, suspension) and implausible ages. Open injuries are kept with
    censored = True; only the Weibull model uses them.
    """
    df = injury_features(sport, as_of)
    keep = ((df["days_missed"] > 0) & (df["days_missed"] <= max_days) & (df["nature"] != "Not an injury")
            & df["age"].between(15, 45))
    df = df[keep].reset_index(drop=True)
    df["injury_class"] = injury_classes(df, min_class_n)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def split(df, test_size=0.2, seed=42):
    """Train/test split by player, so a player's injuries are all on one side."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(df, groups=df["pid"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def completed(df):
    return df[~df["censored"]]


def class_table(df):
    """Each rating class with its size and observed days missed."""
    d = completed(df)
    return (d.groupby("injury_class", observed=True)["days_missed"]
            .agg(injuries="size", mean_days="mean", median_days="median")
            .sort_values("mean_days", ascending=False).round(1))


# --------------------------------------------------------------------------
# Model matrix
# --------------------------------------------------------------------------
class Design:
    """A patsy model matrix that drops columns that are exact combinations of earlier ones.

    Rating classes are nested in body regions, so an interaction such as position x body region
    partly repeats the class columns; patsy can't see that. Columns are checked in order (main effects
    first), so only the redundant interaction columns are dropped and every coefficient is identified.
    Levels absent from the fitting data are dropped the same way.
    """

    def __init__(self, rhs):
        self.rhs = rhs

    def fit(self, df):
        X = dmatrix(self.rhs, df, return_type="dataframe")
        self.info = X.design_info
        A = X.to_numpy(float)
        norms = np.linalg.norm(A, axis=0)
        G = (A.T @ A) / np.outer(np.where(norms > 0, norms, 1), np.where(norms > 0, norms, 1))  # small k x k
        kept = []
        for j in range(A.shape[1]):
            # share of column j not explained by the columns kept so far
            resid = G[j, j] - (G[j, kept] @ np.linalg.solve(G[np.ix_(kept, kept)], G[kept, j]) if kept else 0)
            if norms[j] > 0 and resid > 1e-8:
                kept.append(j)
        keep = np.isin(np.arange(A.shape[1]), kept)
        self.cols = list(X.columns[keep])
        self.dropped = list(X.columns[~keep])
        return X[self.cols]

    def transform(self, df):
        return build_design_matrices([self.info], df, return_type="dataframe")[0][self.cols]


def _gamma_glm(rhs, d):
    design = Design(rhs)
    X = design.fit(d)
    res = sm.GLM(d["days_missed"].to_numpy(), X, family=sm.families.Gamma(sm.families.links.Log())).fit()
    return design, res


# --------------------------------------------------------------------------
# Interaction search
# --------------------------------------------------------------------------

def select_interactions(train, candidates=CANDIDATES, min_gain=0.001, seed=7):
    """Forward selection of interactions for the gamma GLM, judged on a validation split of train
    (by player), so the test set stays untouched.

    Starts from RHS_CLASS and adds, one at a time, the candidate that lowers validation gamma
    deviance the most; stops when no candidate lowers it by at least min_gain (relative).
    Returns (path table, chosen candidate names).
    """
    fit_d, val_d = split(completed(train), test_size=0.25, seed=seed)
    y = val_d["days_missed"]
    def dev(rhs):
        design, res = _gamma_glm(rhs, fit_d)
        out = mean_gamma_deviance(y, res.predict(design.transform(val_d)))
        del design, res
        gc.collect()                      # statsmodels results hold large arrays; free them between trials
        return out

    path = [{"step": "body part + injury type (main effects)", "val_deviance": dev(RHS_MAIN)},
            {"step": "rating classes (body region x injury type, rare ones grouped)", "val_deviance": dev(RHS_CLASS)}]
    chosen, best = [], path[-1]["val_deviance"]
    while True:
        trials = {name: dev(rhs_final(chosen + [name])) for name in candidates if name not in chosen}
        if not trials:
            break
        name, d = min(trials.items(), key=lambda kv: kv[1])
        if d > best * (1 - min_gain):
            break
        chosen.append(name)
        best = d
        path.append({"step": f"+ {name}", "val_deviance": d})
        print(f"  added {name}: validation deviance {d:.4f}", flush=True)
    out = pd.DataFrame(path)
    out["improvement_vs_previous"] = -out["val_deviance"].diff()
    return out.round(5), chosen


# --------------------------------------------------------------------------
# Models. Each has .predict(df) -> expected days; distributional ones also .cdf/.logpdf/.ppf.
# --------------------------------------------------------------------------
class _Distribution:
    def logpdf(self, df):
        return self._dist(df).logpdf(df["days_missed"])

    def cdf(self, df):
        return self._dist(df).cdf(df["days_missed"])

    def ppf(self, df, q):
        return self._dist(df).ppf(q)


class GammaGLM(_Distribution):
    def __init__(self, rhs=None, name="Gamma GLM"):
        self.rhs, self.name = rhs or rhs_final(), name

    def fit(self, train):
        d = completed(train)
        self.design, self.res = _gamma_glm(self.rhs, d)
        y, mu = d["days_missed"].to_numpy(), np.asarray(self.res.fittedvalues)
        # Shape by maximum likelihood (statsmodels' default Pearson scale is unreliable with a heavy tail)
        nll = lambda logk: -stats.gamma.logpdf(y, np.exp(logk), scale=mu / np.exp(logk)).sum()
        self.shape = float(np.exp(optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(self.design.transform(df)))

    def _dist(self, df):
        return stats.gamma(self.shape, scale=self.predict(df) / self.shape)


# --------------------------------------------------------------------------
# Severity tiers: rating classes merged until neighbours really differ
# --------------------------------------------------------------------------
def _tier_rhs(ref, interactions=None):
    rest = rhs_final(interactions).split(" + ", 1)[1]            # everything after the class term
    return f"C(severity_tier, Treatment('{ref}')) + {rest}"


def severity_tiers(train, interactions=None, alpha=0.05, max_rounds=10):
    """Merge rating classes whose relativities are not significantly different into severity tiers.

    The classes are ordered by their gamma GLM relativity (with all the other predictors in the model).
    Neighbouring groups are merged, least different pair first, while the difference between them has
    p > alpha. The model is then refit with the merged groups, and that repeats until every pair of
    neighbouring tiers differs at the alpha level. Fit on training data only: the tiers depend on the
    outcome, so building them on the test set would flatter the test score.

    Returns {rating class: "Tier k"}, Tier 1 being the shortest injuries.
    """
    d = completed(train).copy()
    classes = sorted(d["injury_class"].astype(str).unique())
    group_of = {c: f"G{i}" for i, c in enumerate(classes)}
    for _ in range(max_rounds):
        ref = group_of[REF_CLASS]
        d["severity_tier"] = d["injury_class"].astype(str).map(group_of).astype("category")
        design, res = _gamma_glm(_tier_rhs(ref, interactions), d)
        cov = res.cov_params()
        col = lambda g: f"C(severity_tier, Treatment('{ref}'))[T.{g}]"
        groups = sorted(set(group_of.values()), key=lambda g: 0.0 if g == ref else res.params[col(g)])

        # Each block: estimate b (log relativity vs. the reference group), variance v, member groups
        blocks = [{"b": 0.0 if g == ref else res.params[col(g)], "v": 0.0 if g == ref else cov.loc[col(g), col(g)],
                   "groups": [g]} for g in groups]

        def p_value(a, b):
            c = 0.0
            if len(a["groups"]) == len(b["groups"]) == 1 and ref not in (a["groups"][0], b["groups"][0]):
                c = cov.loc[col(a["groups"][0]), col(b["groups"][0])]
            se = np.sqrt(max(a["v"] + b["v"] - 2 * c, 1e-12))
            return 2 * stats.norm.sf(abs(a["b"] - b["b"]) / se)

        merged = False
        while len(blocks) > 1:
            ps = [p_value(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)]
            i = int(np.argmax(ps))
            if ps[i] <= alpha:
                break
            a, b = blocks[i], blocks[i + 1]
            if ref in a["groups"] + b["groups"]:                      # the reference stays at 0
                new = {"b": 0.0, "v": 0.0}
            else:                                                    # inverse-variance weighted
                w_a, w_b = 1 / a["v"], 1 / b["v"]
                new = {"b": (w_a * a["b"] + w_b * b["b"]) / (w_a + w_b), "v": 1 / (w_a + w_b)}
            blocks[i:i + 2] = [{**new, "groups": a["groups"] + b["groups"]}]
            merged = True
        if not merged:
            break
        rename = {g: blocks[k]["groups"][0] for k in range(len(blocks)) for g in blocks[k]["groups"]}
        group_of = {c: rename[g] for c, g in group_of.items()}

    order = {g: f"Tier {k + 1}" for k, g in enumerate(groups)}      # groups is sorted by relativity
    return {c: order[g] for c, g in group_of.items()}


class TieredGammaGLM(GammaGLM):
    """Gamma GLM with severity tiers in place of rating classes. The tiers are built from the
    training data inside fit()."""

    def __init__(self, interactions=None, name="Gamma GLM, severity tiers + interactions", alpha=0.05):
        self.interactions, self.name, self.alpha = interactions, name, alpha

    def _tiers(self, df):
        out = df.copy()
        out["severity_tier"] = pd.Categorical(out["injury_class"].astype(str).map(self.tier_of),
                                              categories=self.tier_names)
        return out

    def fit(self, train):
        self.tier_of = severity_tiers(train, self.interactions, self.alpha)
        self.tier_names = sorted(set(self.tier_of.values()), key=lambda t: int(t.split()[1]))
        self.ref_tier = self.tier_of[REF_CLASS]
        self.rhs = _tier_rhs(self.ref_tier, self.interactions)
        return super().fit(self._tiers(train))

    def predict(self, df):
        return super().predict(self._tiers(df))

    def tier_table(self, df):
        """One row per tier: multiplier vs. the reference tier, size, observed days, member classes."""
        d = self._tiers(completed(df))
        stats_ = d.groupby("severity_tier", observed=True)["days_missed"].agg(
            injuries="size", mean_days="mean", median_days="median")
        rel = relativities(self)
        prefix = "severity_tier = "
        stats_["multiplier"] = [1.0 if t == self.ref_tier else rel.loc[prefix + t, "multiplier"] for t in stats_.index]
        stats_["ci_low"] = [1.0 if t == self.ref_tier else rel.loc[prefix + t, "ci_low"] for t in stats_.index]
        stats_["ci_high"] = [1.0 if t == self.ref_tier else rel.loc[prefix + t, "ci_high"] for t in stats_.index]
        members = pd.Series(self.tier_of).groupby(pd.Series(self.tier_of)).apply(lambda s: "; ".join(sorted(s.index)))
        stats_["classes"] = members.reindex(stats_.index)
        return stats_.round({"mean_days": 1, "median_days": 1, "multiplier": 3, "ci_low": 3, "ci_high": 3})


class InverseGaussianGLM(GammaGLM):
    def fit(self, train):
        d = completed(train)
        self.design = Design(self.rhs)
        X = self.design.fit(d)
        y = d["days_missed"].to_numpy()
        self.res = sm.GLM(y, X, family=sm.families.InverseGaussian(sm.families.links.Log())).fit()
        mu = np.asarray(self.res.fittedvalues)
        self.lam = float(1 / np.mean((y - mu) ** 2 / (mu ** 2 * y)))      # MLE of the shape parameter
        return self

    def _dist(self, df):
        mu = self.predict(df)
        return stats.invgauss(mu / self.lam, scale=self.lam)


class LognormalModel(_Distribution):
    """Linear regression on log(days); mean on the days scale is exp(xb + sigma^2 / 2)."""

    def __init__(self, rhs=None, name="Lognormal (OLS on log days)"):
        self.rhs, self.name = rhs or rhs_final(), name

    def fit(self, train):
        d = completed(train)
        self.design = Design(self.rhs)
        self.res = sm.OLS(np.log(d["days_missed"].to_numpy()), self.design.fit(d)).fit()
        self.sigma = float(np.sqrt(self.res.ssr / self.res.nobs))
        return self

    def _xb(self, df):
        return np.asarray(self.res.predict(self.design.transform(df)))

    def predict(self, df):
        return np.exp(self._xb(df) + self.sigma ** 2 / 2)

    def _dist(self, df):
        return stats.lognorm(self.sigma, scale=np.exp(self._xb(df)))


class WeibullAFT(_Distribution):
    """Weibull accelerated failure time model. The only one here that uses the censored injuries."""

    def __init__(self, rhs=None, name="Weibull AFT (uses open injuries)"):
        self.rhs, self.name = rhs or rhs_final(), name

    def fit(self, train):
        self.design = Design(self.rhs)
        X = self.design.fit(train).drop(columns="Intercept")      # lifelines adds its own
        self.cols = list(X.columns)
        X["days_missed"] = train["days_missed"].to_numpy()
        X["observed"] = (~train["censored"]).astype(int).to_numpy()
        self.aft = WeibullAFTFitter(penalizer=1e-4).fit(X, "days_missed", "observed")
        lam = self.aft.params_.loc["lambda_"]
        self.beta, self.b0 = lam[self.cols].to_numpy(), float(lam["Intercept"])
        self.rho = float(np.exp(self.aft.params_.loc[("rho_", "Intercept")]))
        return self

    def _scale(self, df):
        X = self.design.transform(df)[self.cols]
        return np.exp(X.to_numpy() @ self.beta + self.b0)

    def predict(self, df):
        return self._scale(df) * special.gamma(1 + 1 / self.rho)

    def _dist(self, df):
        return stats.weibull_min(self.rho, scale=self._scale(df))


class GradientBoosting:
    """Gradient-boosted trees with gamma deviance loss: no formula, finds interactions and
    non-linear effects itself. A benchmark for how much the GLMs leave on the table."""

    def __init__(self, name="Gradient boosting (gamma loss)"):
        self.name = name

    def fit(self, train):
        d = completed(train)
        self.gbm = HistGradientBoostingRegressor(loss="gamma", learning_rate=0.05, max_iter=800,
                                                 early_stopping=True, categorical_features="from_dtype",
                                                 random_state=0)
        self.gbm.fit(d[FEATURES], d["days_missed"])
        return self

    def predict(self, df):
        return self.gbm.predict(df[FEATURES])


def default_models(interactions=None):
    final = rhs_final(interactions)
    return [
        GammaGLM("1", "Gamma, no predictors"),
        GammaGLM(RHS_MAIN, "Gamma GLM, main effects"),
        GammaGLM(final, "Gamma GLM, rating classes + interactions"),
        TieredGammaGLM(interactions),
        InverseGaussianGLM(final, "Inverse Gaussian GLM"),
        LognormalModel(final, "Lognormal (OLS on log days)"),
        WeibullAFT(final, "Weibull AFT (uses open injuries)"),
        GradientBoosting(),
    ]


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
    """Score every model on the held-out completed injuries.

    log_lik      average log-likelihood per injury (higher is better) — the fairest comparison of
                 whole distributions, which is what pricing needs
    gamma_dev    mean gamma deviance (lower is better) — the gamma GLM's own loss
    mae / rmse   error in days on the expected value
    above_p90    share of injuries longer than the model's 90th percentile; a calibrated tail gives 10%
    """
    d = completed(test)
    y = d["days_missed"].to_numpy()
    rows = []
    for name, m in models.items():
        mu = m.predict(d)
        dist = hasattr(m, "logpdf")
        rows.append({
            "model": name,
            "log_lik": np.mean(m.logpdf(d)) if dist else np.nan,
            "gamma_dev": mean_gamma_deviance(y, mu),
            "mae": np.mean(np.abs(y - mu)),
            "rmse": np.sqrt(np.mean((y - mu) ** 2)),
            "mean_pred": mu.mean(),
            "above_p90": np.mean(y > m.ppf(d, 0.9)) if dist else np.nan,
        })
    out = pd.DataFrame(rows).set_index("model")
    out.loc["(actual)", "mean_pred"] = y.mean()
    return out.round({"log_lik": 4, "gamma_dev": 4, "mae": 2, "rmse": 2, "mean_pred": 2, "above_p90": 4})


def calibration(models, test, bins=10):
    """Average predicted vs. actual days, by decile of each model's prediction."""
    d = completed(test)
    rows = []
    for name, m in models.items():
        mu = m.predict(d)
        if np.ptp(mu) == 0:          # constant prediction (no-predictor model): nothing to bin
            continue
        dec = pd.qcut(mu, bins, labels=False, duplicates="drop")
        g = pd.DataFrame({"pred": mu, "actual": d["days_missed"].to_numpy(), "decile": dec}).groupby("decile")
        rows.append(g.mean().assign(model=name, n=g.size()))
    return pd.concat(rows).reset_index()


def tidy_terms(index):
    """Readable coefficient names: 'C(region)[T.Knee]' -> 'region = Knee', ':' -> ' x '."""
    return (pd.Index(index)
            .str.replace(r"C\((\w+)(?:, Treatment\('.*?'\))?\)\[T\.(.*?)\]", r"\1 = \2", regex=True)
            .str.replace(r"np\.log1p\((\w+)\)", r"log(1 + \1)", regex=True)
            .str.replace(":", " x ", regex=False))


def relativities(model):
    """exp(coefficient) with 95% CI: how much each factor multiplies expected days missed."""
    res = model.res
    ci = np.exp(res.conf_int())
    out = pd.DataFrame({"multiplier": np.exp(res.params), "ci_low": ci[0], "ci_high": ci[1],
                        "p_value": res.pvalues})
    out.index = tidy_terms(out.index)
    return out.round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="Soccer")
    p.add_argument("--max-days", type=int, default=1000)
    p.add_argument("--select", action="store_true", help="re-run the interaction search instead of FINAL_INTERACTIONS")
    args = p.parse_args()
    pd.set_option("display.width", 160)

    data = build_dataset(args.sport, args.max_days)
    train, test = split(data)
    print(f"{len(data):,} injuries ({data['censored'].sum():,} still open), {data['pid'].nunique():,} players, "
          f"{data['injury_class'].nunique()} rating classes; train {len(train):,} / test {len(test):,}")
    interactions = None
    if args.select:
        path, interactions = select_interactions(train)
        print(path.to_string())
    models = fit_all(train, default_models(interactions))
    print("\nHeld-out comparison (completed test injuries):")
    print(compare(models, test).to_string())
    print("\nGamma GLM (rating classes + interactions) relativities:")
    print(relativities(models["Gamma GLM, rating classes + interactions"]).to_string())


if __name__ == "__main__":
    main()
