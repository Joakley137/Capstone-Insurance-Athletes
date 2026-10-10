"""
backtest.py — season-level back-test of the pricing model on held-out players (used by
pricing_model.ipynb and make_figures.py).

    from runs import backtest
    bt = backtest.run()              # one row per held-out big-five player-season: lam, q_D, c, predictions, actuals

Fitting (pricing.fit) uses training players only: the frequency NB GLM on frequency-train player-seasons,
the severity GLM and the injury mix on severity-train players' injuries in those big-five player-seasons
(pricing.insured_injuries), the career GLM on all-league injuries of training players (career rows of any
frequency-test player are dropped too, since career.split is not aligned with the other two).
Each held-out player-season is described by what is known at the season start (league, age, position, minutes
in the last 12 months, career minutes, prior injuries, date = season start); the per-injury models are
averaged over the injury mix exactly as pricing.injury_probs does, but for all rows at once.
"""
import numpy as np
import pandas as pd

from model import career
from model import frequency as freq
from model import predict
from model import pricing
from model import severity as sev

DEFERMENTS = (30, 60, 90, 180)


def fit_models():
    """Training-player fits -> (nb, bundle, data dict)."""
    fd = freq.build_dataset()
    f_train, f_test = freq.split(fd)
    sd = sev.build_dataset()
    s_train, s_test = sev.split(sd)
    cd = career.build_dataset()
    held_out = set(s_test["pid"]) | set(f_test["pid"])
    c_train = cd[~cd["pid"].isin(held_out)].reset_index(drop=True)
    b = pricing.fit(s_train, c_train, f_train)
    return b["frequency"], b, {"freq": fd, "f_train": f_train, "f_test": f_test, "career": cd, "c_train": c_train}


def season_inputs(seasons):
    """Per-injury model inputs known at each season start (the quote's player inputs)."""
    f = sev.injury_features()
    ps = sev.player_seasons().sort_values(["ext_player_id", "season_end"])
    ps["cum_minutes"] = ps.groupby("ext_player_id")["minutes"].cumsum()
    d = seasons[["pid", "season_start", "league", "age", "position_group", "minutes_last_season"]].copy()
    d["league"] = d["league"].astype(str)
    d["_i"] = np.arange(len(d))
    d = d.sort_values("season_start")
    # prior_injuries as the severity model counts them: every recorded injury before the date
    f = f.sort_values("injury_date").assign(n_before=lambda x: x.groupby("pid").cumcount() + 1)
    hist = pd.merge_asof(d, f[["pid", "injury_date", "n_before"]], left_on="season_start", right_on="injury_date",
                         by="pid", direction="backward", allow_exact_matches=False)
    cum = pd.merge_asof(d, ps[["ext_player_id", "season_end", "cum_minutes"]].sort_values("season_end"),
                        left_on="season_start", right_on="season_end", left_by="pid", right_by="ext_player_id",
                        direction="backward", allow_exact_matches=False)
    d["prior_injuries"] = hist["n_before"].fillna(0).to_numpy()
    d["career_games"] = cum["cum_minutes"].fillna(0).to_numpy() / 90
    d = d.sort_values("_i").drop(columns="_i").reset_index(drop=True)
    pos = d["position_group"].astype(str)
    d["position_group"] = pos.where(pos.isin(predict.POSITIONS), "Midfield")   # quote() default
    d["games_12m"] = d["minutes_last_season"] / 90
    d["no_games_12m"] = (d["games_12m"] <= 0).astype(int)
    d["age_c"] = d["age"] - 26
    d["reinjury_60d"] = d["prior_same_part"] = 0                                 # quote() defaults
    d["year"] = d["season_start"].dt.year
    d["year_c"] = d["year"] - 2015
    for p in ["Attack", "Defender", "Goalkeeper"]:
        d[f"pos_{p.lower()}"] = (d["position_group"] == p).astype(int)
    return d


def mix_probs(b, inputs, deferments=DEFERMENTS):
    """q_D and c per row, averaged over the injury mix (vectorised pricing.injury_probs)."""
    mix = b["mix"]
    n, k = len(inputs), len(mix)
    rows = inputs.loc[inputs.index.repeat(k)].reset_index(drop=True)
    rows["region"] = np.tile(mix["region"].to_numpy(), n)
    rows["nature"] = np.tile(mix["nature"].to_numpy(), n)
    cls = [predict._rating_class(b, r, t) for r, t in zip(mix["region"], mix["nature"])]
    rows["injury_class"] = np.tile(cls, n)
    w = mix["w"].to_numpy()
    out = pd.DataFrame(index=inputs.index)
    for dd in deferments:
        out[f"q_{dd}"] = pricing._tail(b["tiered"], rows, dd)[0].reshape(n, k) @ w
    out["c"] = pricing._career_p(b, rows).reshape(n, k) @ w
    return out


def actuals(test, career_data):
    """Observed per player-season: any injury, any injury > D days, career-ending injury (labelled seasons)."""
    inj = freq.covered_injuries()
    out = pd.DataFrame(index=test.index)
    out["any_injury"] = (test["n_injuries"] > 0).astype(int)
    key = pd.MultiIndex.from_frame(test[["pid", "season"]])
    for d in DEFERMENTS:
        # open injuries count with days so far; all seasons here ended > 1 year before the data date
        n = inj[inj["days_missed"] > d].groupby(["pid", "season"]).size()
        out[f"over_{d}"] = (n.reindex(key).fillna(0).to_numpy() > 0).astype(int)
    ce = career_data[career_data["career_ending"] == 1]
    ce_season = ce["injury_date"].dt.year - (ce["injury_date"].dt.month < 7)
    ce_n = ce.groupby([ce["pid"], ce_season.rename("season")]).size()
    out["career_ending"] = (ce_n.reindex(key).fillna(0).to_numpy() > 0).astype(int)
    last = career_data.attrs["cutoff"].year - 1           # last season that ends before the label cut-off
    out["career_labelled"] = test["season"].le(last).to_numpy()
    return out


def run():
    nb, b, data = fit_models()
    test = data["f_test"]
    lam = nb.predict(test)
    inputs = season_inputs(test)
    probs = mix_probs(b, inputs)
    bt = pd.concat([test[["pid", "season", "league", "position_group", "age", "minutes_last_season", "n_injuries"]],
                    inputs[["prior_injuries", "career_games"]], probs], axis=1)
    bt["lam"] = lam
    bt["p_any"] = pricing.p_any(lam, nb.alpha)
    for d in DEFERMENTS:
        bt[f"p_covered_{d}"] = pricing.p_any(lam * bt[f"q_{d}"], nb.alpha)
    bt["p_career"] = pricing.p_any(lam * bt["c"], nb.alpha)
    bt = pd.concat([bt, actuals(test, data["career"])], axis=1)
    bt.attrs.update(alpha=nb.alpha, nb=nb, bundle=b, n_train=len(data["f_train"]),
                    n_c_train=len(data["c_train"]), n_s_train=b["n_insured_injuries"])
    return bt


def by_group(bt, pred, actual, bins=10):
    """Mean predicted vs actual share by predicted-risk group (1 = lowest)."""
    g = pd.qcut(bt[pred].rank(method="first"), bins, labels=range(1, bins + 1))
    out = bt.groupby(g, observed=True).agg(n=(pred, "size"), predicted=(pred, "mean"), actual=(actual, "mean"))
    out.index.name = "decile" if bins == 10 else "quintile" if bins == 5 else "group"
    return out


def capture(bt, pred, actual, top=0.2):
    """Share of actual events among the riskiest `top` share of rows."""
    cut = bt[pred].rank(method="first", ascending=False) <= top * len(bt)
    return bt.loc[cut, actual].sum() / bt[actual].sum()
