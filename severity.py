"""
severity.py — injury severity (days missed): gamma GLM on severity tiers.

    python severity.py                        # fit, print held-out comparison and coefficients

    import severity as sev
    train, test = sev.split(sev.build_dataset())
    models = sev.fit_all(train)
    sev.compare(models, test)
"""
import argparse
import functools
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy import optimize, stats
from sklearn.metrics import mean_gamma_deviance
from sklearn.model_selection import GroupShuffleSplit

import sportsdb as sdb

# Injury type from the description; first match wins
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

# Rare body parts merged with neighbours
REGIONS = {"Quadriceps": "Thigh", "Back": "Back / neck", "Neck": "Back / neck", "Hand": "Arm / hand",
           "Wrist": "Arm / hand", "Elbow": "Arm / hand", "Chest": "Trunk", "Ribs": "Trunk", "Core": "Trunk"}

MIN_CLASS_N = 300            # min injuries for a region x type cell to be its own class
REF_CLASS = "Hamstring (unspecified)"

CATEGORICAL = ["body_part", "region", "nature", "injury_class", "position_group"]

# Player terms. Position as 0/1 dummies (Midfield = base).
RHS_PLAYER = ("pos_attack + pos_defender + pos_goalkeeper + age_c + I(age_c ** 2)"
              " + np.log1p(prior_injuries) + reinjury_60d + prior_same_part")
RHS_CLASS = f"C(injury_class, Treatment('{REF_CLASS}')) + " + RHS_PLAYER


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
    """Minutes and appearances per player-season, all competitions."""
    ps = sdb.query("""SELECT ext_player_id, season_start, season_end,
                             SUM(minutes) AS minutes, SUM(appearances) AS appearances
                      FROM player_seasons WHERE sport = :sport AND season_end IS NOT NULL
                      GROUP BY ext_player_id, season_start, season_end""", {"sport": sport})
    for c in ["season_start", "season_end"]:
        ps[c] = pd.to_datetime(ps[c])
    return ps


def _asof(left, right, on_left, on_right, value, by="ext_player_id"):
    """Latest `value` from right (same player) with on_right <= on_left."""
    l = left[["_row", by, on_left]].dropna().sort_values(on_left)
    r = right[[by, on_right, value]].sort_values(on_right)
    hit = pd.merge_asof(l, r, left_on=on_left, right_on=on_right, by=by, direction="backward")
    return left["_row"].map(hit.set_index("_row")[value])


def injury_features(sport="Soccer", as_of=None):
    """All injury events with predictors, unfiltered. Open injuries: censored=True, days_missed = days so far."""
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

    # Injury history (prior injuries only)
    g = df.groupby("pid")
    df["prior_injuries"] = g.cumcount()
    since_last = df["injury_date"] - g["return_date"].shift()
    df["reinjury_60d"] = (since_last <= pd.to_timedelta(60, unit="D")).astype(int)
    df["prior_same_part"] = (df["body_part"].notna()
                             & (df.groupby(["pid", "body_part"]).cumcount() > 0)).astype(int)

    # Playing time: only seasons ended before the injury (avoids leakage)
    ps = player_seasons(sport)
    df["games_12m"] = df["career_games"] = 0.0
    if not ps.empty:
        ps = ps.sort_values(["ext_player_id", "season_end"])
        ps["cum_minutes"] = ps.groupby("ext_player_id")["minutes"].cumsum()
        df["_minus_12m"] = df["injury_date"] - pd.to_timedelta(365, unit="D")
        career_now = _asof(df, ps, "injury_date", "season_end", "cum_minutes").fillna(0)
        career_12m_ago = _asof(df, ps, "_minus_12m", "season_end", "cum_minutes").fillna(0)
        df["career_games"] = career_now / 90                   # full-game equivalents
        df["games_12m"] = (career_now - career_12m_ago) / 90
        df = df.drop(columns="_minus_12m")

    df["body_part"] = df["body_part"].fillna("Unknown")
    df["region"] = df["body_part"].replace(REGIONS)
    df["nature"] = df["injury_desc"].map(nature)
    df["position_group"] = df["position_group"].fillna("Unknown")
    df["year"] = df["injury_date"].dt.year
    df["offseason"] = df["injury_date"].dt.month.isin([6, 7]).astype(int)

    # Centered for the formula models
    df["age_c"] = df["age"] - 26
    df["year_c"] = df["year"] - 2015
    df["no_games_12m"] = (df["games_12m"] <= 0).astype(int)
    for pos in ["Attack", "Defender", "Goalkeeper"]:
        df[f"pos_{pos.lower()}"] = (df["position_group"] == pos).astype(int)
    return df


def injury_classes(df, min_n=MIN_CLASS_N):
    """Rating classes: region x type if >= min_n, else pooled by type, then region, then "Other (rare)".
    Uses counts only (no outcome), so no test leakage."""
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
    """Severity modeling table: one row per injury, 0 < days <= max_days, real injuries, ages 15-45."""
    df = injury_features(sport, as_of)
    keep = ((df["days_missed"] > 0) & (df["days_missed"] <= max_days) & (df["nature"] != "Not an injury")
            & df["age"].between(15, 45))
    df = df[keep].reset_index(drop=True)
    df["injury_class"] = injury_classes(df, min_class_n)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def split(df, test_size=0.2, seed=42):
    """Train/test split grouped by player."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(df, groups=df["pid"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def completed(df):
    return df[~df["censored"]]


# --------------------------------------------------------------------------
# Model matrix
# --------------------------------------------------------------------------
class Design:
    """Patsy model matrix that drops columns linearly dependent on earlier ones."""

    def __init__(self, rhs):
        self.rhs = rhs

    def fit(self, df):
        X = dmatrix(self.rhs, df, return_type="dataframe")
        self.info = X.design_info
        self._template = df.iloc[:1].copy()
        A = X.to_numpy(float)
        norms = np.linalg.norm(A, axis=0)
        G = (A.T @ A) / np.outer(np.where(norms > 0, norms, 1), np.where(norms > 0, norms, 1))  # small k x k
        kept = []
        for j in range(A.shape[1]):
            resid = G[j, j] - (G[j, kept] @ np.linalg.solve(G[np.ix_(kept, kept)], G[kept, j]) if kept else 0)
            if norms[j] > 0 and resid > 1e-8:
                kept.append(j)
        keep = np.isin(np.arange(A.shape[1]), kept)
        self.cols = list(X.columns[keep])
        self.dropped = list(X.columns[~keep])
        return X[self.cols]

    def transform(self, df):
        return build_design_matrices([self.info], df, return_type="dataframe")[0][self.cols]

    # design_info isn't picklable; rebuilt on load
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "info"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.info = dmatrix(self.rhs, self._template, return_type="dataframe").design_info


# ==================================================================
# GLM FIT — gamma GLM, log link
# ==================================================================
def _gamma_glm(rhs, d):
    design = Design(rhs)
    X = design.fit(d)
    res = sm.GLM(d["days_missed"].to_numpy(), X, family=sm.families.Gamma(sm.families.links.Log())).fit()
    return design, res


# ==================================================================
# GLM FIT — GammaGLM model class (fits via _gamma_glm, shape by MLE)
# ==================================================================
class GammaGLM:
    """.predict(df) -> expected days; .logpdf/.ppf from the fitted gamma."""

    def __init__(self, rhs=RHS_CLASS, name="Gamma GLM"):
        self.rhs, self.name = rhs, name

    def fit(self, train):
        d = completed(train)
        self.design, self.res = _gamma_glm(self.rhs, d)
        y, mu = d["days_missed"].to_numpy(), np.asarray(self.res.fittedvalues)
        # Shape by MLE (Pearson scale is unreliable with a heavy tail)
        nll = lambda logk: -stats.gamma.logpdf(y, np.exp(logk), scale=mu / np.exp(logk)).sum()
        self.shape = float(np.exp(optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(self.design.transform(df)))

    def _dist(self, df):
        return stats.gamma(self.shape, scale=self.predict(df) / self.shape)

    def logpdf(self, df):
        return self._dist(df).logpdf(df["days_missed"])

    def ppf(self, df, q):
        return self._dist(df).ppf(q)


# --------------------------------------------------------------------------
# Severity tiers
# --------------------------------------------------------------------------
def _tier_rhs(ref):
    return f"C(severity_tier, Treatment('{ref}')) + " + RHS_PLAYER


def severity_tiers(train, alpha=0.05, max_rounds=10):
    """Merge adjacent rating classes (by relativity) until neighbours differ at alpha; refit each round.
    Train data only. Returns {rating class: "Tier k"}, Tier 1 = shortest."""
    d = completed(train).copy()
    classes = sorted(d["injury_class"].astype(str).unique())
    group_of = {c: f"G{i}" for i, c in enumerate(classes)}
    for _ in range(max_rounds):
        ref = group_of[REF_CLASS]
        d["severity_tier"] = d["injury_class"].astype(str).map(group_of).astype("category")
        design, res = _gamma_glm(_tier_rhs(ref), d)
        cov = res.cov_params()
        col = lambda g: f"C(severity_tier, Treatment('{ref}'))[T.{g}]"
        groups = sorted(set(group_of.values()), key=lambda g: 0.0 if g == ref else res.params[col(g)])

        # block: log relativity b, variance v, member groups
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
            if ref in a["groups"] + b["groups"]:                      # reference stays at 0
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

    order = {g: f"Tier {k + 1}" for k, g in enumerate(groups)}
    return {c: order[g] for c, g in group_of.items()}


class TieredGammaGLM(GammaGLM):
    """Gamma GLM on severity tiers (built in fit()) instead of rating classes."""

    def __init__(self, name="Gamma GLM, severity tiers", alpha=0.05):
        self.name, self.alpha = name, alpha

    def _tiers(self, df):
        out = df.copy()
        out["severity_tier"] = pd.Categorical(out["injury_class"].astype(str).map(self.tier_of),
                                              categories=self.tier_names)
        return out

    def fit(self, train):
        self.tier_of = severity_tiers(train, self.alpha)
        self.tier_names = sorted(set(self.tier_of.values()), key=lambda t: int(t.split()[1]))
        self.ref_tier = self.tier_of[REF_CLASS]
        self.rhs = _tier_rhs(self.ref_tier)
        super().fit(self._tiers(train))
        # Per-tier empirical quantiles of actual / predicted
        d = completed(train)
        ratio = d["days_missed"].to_numpy() / self.predict(d)
        tier = d["injury_class"].astype(str).map(self.tier_of).to_numpy()
        grid = np.linspace(0, 1, 1001)
        self.ratio_quantiles = {t: np.quantile(ratio[tier == t], grid) for t in self.tier_names}
        return self

    def predict(self, df):
        return super().predict(self._tiers(df))

    def _ratio_q(self, df):
        tiers = df["injury_class"].astype(str).map(self.tier_of).to_numpy()
        return np.vstack([self.ratio_quantiles[t] for t in tiers])

    def range_ppf(self, df, q):
        """q-th quantile of days."""
        rq = self._ratio_q(df)
        k = int(round(q * (rq.shape[1] - 1)))
        return self.predict(df) * rq[:, k]

    def range_sf(self, df, days):
        """P(days missed > days)."""
        rq, mu = self._ratio_q(df), self.predict(df)
        return np.array([1 - np.searchsorted(r, days / m, side="right") / len(r) for r, m in zip(rq, mu)])


def default_models():
    return [GammaGLM("1", "Gamma, no predictors"), TieredGammaGLM()]


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
    """Score models on completed test injuries: log_lik (higher better), gamma_dev, mae, rmse,
    above_p90 (calibrated = 0.10)."""
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


def tidy_terms(index):
    """Readable coefficient names: 'C(region)[T.Knee]' -> 'region = Knee', ':' -> ' x '."""
    return (pd.Index(index)
            .str.replace(r"C\((\w+)(?:, Treatment\('.*?'\))?\)\[T\.(.*?)\]", r"\1 = \2", regex=True)
            .str.replace(r"np\.log1p\((\w+)\)", r"log(1 + \1)", regex=True)
            .str.replace(":", " x ", regex=False))


def coefficients(model):
    """Fitted GLM coefficients (link scale) and p-values, with readable term names."""
    res = model.res
    out = pd.DataFrame({"coef": res.params, "p_value": res.pvalues})
    out.index = tidy_terms(out.index).rename("variable")
    return out.round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="Soccer")
    p.add_argument("--max-days", type=int, default=1000)
    args = p.parse_args()
    pd.set_option("display.width", 160)

    data = build_dataset(args.sport, args.max_days)
    train, test = split(data)
    print(f"{len(data):,} injuries ({data['censored'].sum():,} still open), {data['pid'].nunique():,} players, "
          f"{data['injury_class'].nunique()} rating classes; train {len(train):,} / test {len(test):,}")
    models = fit_all(train)
    tiered = models["Gamma GLM, severity tiers"]
    print("\nHeld-out comparison (completed test injuries):")
    print(compare(models, test).to_string())
    print(f"\nGamma GLM ({len(tiered.tier_names)} severity tiers) coefficients:")
    print(coefficients(tiered).to_string())

if __name__ == "__main__":
    main()
