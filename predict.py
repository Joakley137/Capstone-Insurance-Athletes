"""
predict.py — predictions for injuries you describe.

    python predict.py --injury "Cruciate ligament tear" --age 27 --position Defender \\
                      --market-value 15000000 --minutes-last-12-months 2500
    python predict.py --csv my_injuries.csv              # one prediction per row -> printed / --out file.csv
    python predict.py --refit                            # refit the saved models (after reloading data)

    import predict
    predict.injury("Cruciate ligament tear", age=27, position="Defender", market_value=15e6)
    predict.from_csv("my_injuries.csv")

You get:
  * expected days out (gamma GLM with severity tiers) and the severity tier,
  * a realistic range — median, 75th and 90th percentile, chance of missing more than 90 / 180 days —
    from how actual injuries in the same tier spread around the model's prediction (checked on held-out
    players: about 50% fall below the predicted median and 10% above the 90th percentile, in every tier),
  * the probability the injury ends the player's career (logistic GLM).

The injury is described in words, like the Transfermarkt descriptions ("Hamstring strain", "Broken
foot"), or with body_part= and injury_type= directly. Anything you leave out is set to a typical value
(the median player in the data) and listed under `assumed`, so you can see what the prediction rests on.

The models are fitted once on all the soccer data and saved to models/injury_models.pkl; that first
run takes a couple of minutes, later ones are instant. Predictions are for professional soccer.
"""
import argparse
import datetime as dt
import os
import pickle

import numpy as np
import pandas as pd

import build_db
import career
import severity as sev

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "injury_models.pkl")
POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
INJURY_TYPES = sorted({label for _, label in sev.NATURE if label != "Not an injury"} | {"Unspecified"})
BODY_PARTS = sorted({part for _, part in build_db.BODY_PARTS} | {"Unknown"})


# --------------------------------------------------------------------------
# Fitting and saving
# --------------------------------------------------------------------------
def _slim(model):
    """Drop the per-row training arrays statsmodels keeps, so the saved file stays small."""
    model.res.remove_data()
    return model


def fit_and_save(path=MODEL_FILE):
    print("Fitting the models on all the soccer data (a couple of minutes, once) ...", flush=True)
    data = sev.build_dataset()
    tiered = _slim(sev.TieredGammaGLM().fit(data))
    cdata = career.build_dataset()
    car = _slim(career.LogisticGLM(career.RHS_MAIN).fit(cdata))

    cells = data.groupby(["region", "nature"], observed=True)["injury_class"].first().astype(str)
    bundle = {
        "tiered": tiered, "career": car,
        "class_of": {(r, n): c for (r, n), c in cells.items()},
        "classes": sorted(data["injury_class"].astype(str).unique()),
        "typical": {"prior_injuries": float(data["prior_injuries"].median()),
                    "market_value": float(np.exp(data["log_mv"].median())),
                    "games_12m": float(data["games_12m"].median()),
                    "career_games": float(data["career_games"].median())},
        "career_last_year": int(cdata["year"].max()),
        "fitted": dt.date.today().isoformat(), "n_injuries": len(data), "n_career": len(cdata),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved to {path}")
    return bundle


_bundle = None


def models(refit=False):
    global _bundle
    if refit or (_bundle is None and not os.path.exists(MODEL_FILE)):
        _bundle = fit_and_save()
    elif _bundle is None:
        with open(MODEL_FILE, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


# --------------------------------------------------------------------------
# Building the feature row
# --------------------------------------------------------------------------
def _rating_class(b, region, nature):
    """Same rule as severity.injury_classes: the combination's own class if it has one, otherwise
    the pooled class for that injury type, then for that body region, then 'Other (rare)'."""
    if (region, nature) in b["class_of"]:
        return b["class_of"][(region, nature)]
    by_type = ("Unspecified injury" if nature == "Unspecified" else nature) + " - other sites"
    for c in [by_type, f"{region} - other types", "Other (rare)"]:
        if c in b["classes"]:
            return c
    return sev.REF_CLASS


def _row(b, injury=None, age=None, position=None, body_part=None, injury_type=None, market_value=None,
         minutes_last_12_months=None, career_minutes=None, prior_injuries=None, same_body_part_before=False,
         returned_within_60_days=False, date=None):
    if age is None:
        raise ValueError("age is required")
    assumed = []
    part = body_part or build_db.body_part(injury) or "Unknown"
    kind = injury_type or sev.nature(injury)
    if kind == "Not an injury":
        raise ValueError(f"'{injury}' reads as rest/fitness/suspension, not an injury")
    if part not in BODY_PARTS:
        raise ValueError(f"body_part must be one of {BODY_PARTS}")
    if kind not in INJURY_TYPES:
        raise ValueError(f"injury_type must be one of {INJURY_TYPES}")
    if position is None:
        position = "Midfield"
        assumed.append("position = Midfield")
    position = position.strip().title()
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")

    t = b["typical"]
    if prior_injuries is None:
        prior_injuries = t["prior_injuries"]
        assumed.append(f"prior injuries = {prior_injuries:.0f}")
    games_12m = t["games_12m"] if minutes_last_12_months is None else minutes_last_12_months / 90
    if minutes_last_12_months is None:
        assumed.append(f"minutes in last 12 months = {games_12m * 90:,.0f}")
    career_games = t["career_games"] if career_minutes is None else career_minutes / 90
    if career_minutes is None:
        assumed.append(f"career minutes = {career_games * 90:,.0f}")
    if market_value is None:
        market_value = t["market_value"]
        assumed.append(f"market value = EUR {market_value:,.0f}")
    elif market_value <= 0:      # 0 = explicitly no market value: a player outside the valued leagues
        market_value = None
    when = pd.Timestamp(date) if date else pd.Timestamp(dt.date.today())

    region = sev.REGIONS.get(part, part)
    row = {
        "body_part": part, "region": region, "nature": kind,
        "injury_class": _rating_class(b, region, kind), "position_group": position,
        "age": float(age), "age_c": float(age) - 26,
        "prior_injuries": float(prior_injuries), "reinjury_60d": int(bool(returned_within_60_days)),
        "prior_same_part": int(bool(same_body_part_before)),
        "log_mv": np.log(market_value) if market_value else np.nan,
        "log_mv_c": np.log(market_value) - sev.LOG_MV_CENTER if market_value else 0.0,
        "mv_missing": int(not market_value),
        "games_12m": float(games_12m), "no_games_12m": int(games_12m <= 0), "career_games": float(career_games),
        "year": when.year, "year_c": when.year - 2015, "offseason": int(when.month in (6, 7)),
    }
    for pos in ["Attack", "Defender", "Goalkeeper"]:
        row[f"pos_{pos.lower()}"] = int(position == pos)
    return row, assumed


# --------------------------------------------------------------------------
# Predicting
# --------------------------------------------------------------------------
def _predict_frame(b, rows):
    df = pd.DataFrame(rows)
    tiered, car = b["tiered"], b["career"]
    out = pd.DataFrame(index=df.index)
    out["body_part"], out["injury_type"], out["rating_class"] = df["body_part"], df["nature"], df["injury_class"]
    out["severity_tier"] = df["injury_class"].map(tiered.tier_of)
    out["expected_days"] = tiered.predict(df).round(1)
    out["median_days"] = tiered.range_ppf(df, 0.5).round(0)
    out["p75_days"] = tiered.range_ppf(df, 0.75).round(0)
    out["p90_days"] = tiered.range_ppf(df, 0.9).round(0)
    out["chance_over_90_days"] = tiered.range_sf(df, 90).round(3)
    out["chance_over_180_days"] = tiered.range_sf(df, 180).round(3)
    # The career data ends earlier; its year trend mostly reflects data coverage, so don't extrapolate it
    cdf = df.assign(year_c=np.minimum(df["year"], b["career_last_year"]) - 2015)
    for c in ["region", "nature", "position_group"]:
        cdf[c] = cdf[c].astype(str)
    out["career_ending_chance"] = car.predict(cdf).round(4)
    return out


def injury(injury=None, *, age, position=None, body_part=None, injury_type=None, market_value=None,
           minutes_last_12_months=None, career_minutes=None, prior_injuries=None, same_body_part_before=False,
           returned_within_60_days=False, date=None):
    """Predict one injury. Returns a pandas Series; `assumed` lists the inputs that were filled in."""
    b = models()
    row, assumed = _row(b, injury, age, position, body_part, injury_type, market_value, minutes_last_12_months,
                        career_minutes, prior_injuries, same_body_part_before, returned_within_60_days, date)
    out = _predict_frame(b, [row]).iloc[0]
    out["assumed"] = "; ".join(assumed) or "nothing"
    return out


CSV_COLUMNS = ["injury", "age", "position", "body_part", "injury_type", "market_value", "minutes_last_12_months",
               "career_minutes", "prior_injuries", "same_body_part_before", "returned_within_60_days", "date"]


def from_csv(path):
    """One prediction per row of a CSV with any of the CSV_COLUMNS (only `age`, and `injury` or
    `body_part`, are required). Extra `name` / `player` columns are passed through."""
    b = models()
    raw = pd.read_csv(path)
    unknown = set(raw.columns) - set(CSV_COLUMNS) - {"name", "player"}
    if unknown:
        raise ValueError(f"Unknown columns {sorted(unknown)}; expected some of {CSV_COLUMNS}")
    rows, notes = [], []
    for _, r in raw.iterrows():
        args = {k: (None if pd.isna(v) else v) for k, v in r.items() if k in CSV_COLUMNS}
        for flag in ["same_body_part_before", "returned_within_60_days"]:
            v = args.get(flag)
            args[flag] = str(v).strip().lower() in ("1", "true", "yes", "y") if v is not None else False
        row, assumed = _row(b, **args)
        rows.append(row)
        notes.append("; ".join(assumed) or "nothing")
    out = _predict_frame(b, rows)
    out["assumed"] = notes
    return pd.concat([raw, out], axis=1)


def explain(p):
    """A short plain-language read-out of one prediction from injury()."""
    return "\n".join([
        f"Read as: {p.injury_type} — {p.body_part}  (rating class '{p.rating_class}', "
        f"{p.severity_tier} of {len(models()['tiered'].tier_names)}, higher = longer)",
        f"Expected time out: {p.expected_days:.0f} days",
        f"Likely range: half of similar injuries take under {p.median_days:.0f} days, "
        f"3 in 4 under {p.p75_days:.0f}, 9 in 10 under {p.p90_days:.0f}",
        f"Chance of missing more than 90 days: {p.chance_over_90_days:.0%};  more than 180 days: "
        f"{p.chance_over_180_days:.0%}",
        f"Chance this injury ends the career: "
        f"{'under 0.01%' if p.career_ending_chance < 0.0001 else f'{p.career_ending_chance:.2%}'}"
        f"  (average injury: about 0.42%)",
        f"Filled in with typical values: {p.assumed}",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--injury", help='description, e.g. "Hamstring strain", "Broken foot"')
    p.add_argument("--age", type=float)
    p.add_argument("--position", choices=POSITIONS)
    p.add_argument("--body-part", choices=BODY_PARTS)
    p.add_argument("--injury-type", choices=INJURY_TYPES)
    p.add_argument("--market-value", type=float, help="euros; 0 = no market value (lower leagues)")
    p.add_argument("--minutes-last-12-months", type=float)
    p.add_argument("--career-minutes", type=float)
    p.add_argument("--prior-injuries", type=int)
    p.add_argument("--same-body-part-before", action="store_true")
    p.add_argument("--returned-within-60-days", action="store_true", help="hurt again within 60 days of returning")
    p.add_argument("--date", help="injury date, YYYY-MM-DD (default today)")
    p.add_argument("--csv", help="predict every row of this CSV instead")
    p.add_argument("--out", help="with --csv: write the predictions here")
    p.add_argument("--refit", action="store_true", help="refit and re-save the models")
    a = p.parse_args()

    if a.refit:
        models(refit=True)
        if not (a.csv or a.injury or a.body_part):
            return
    if a.csv:
        res = from_csv(a.csv)
        if a.out:
            res.to_csv(a.out, index=False)
            print(f"Wrote {len(res)} predictions to {a.out}")
        else:
            pd.set_option("display.width", 200)
            print(res.to_string(index=False))
        return
    if a.age is None or not (a.injury or a.body_part):
        p.error("give --age and --injury (or --body-part), or --csv FILE")
    res = injury(a.injury, age=a.age, position=a.position, body_part=a.body_part, injury_type=a.injury_type,
                 market_value=a.market_value, minutes_last_12_months=a.minutes_last_12_months,
                 career_minutes=a.career_minutes, prior_injuries=a.prior_injuries,
                 same_body_part_before=a.same_body_part_before,
                 returned_within_60_days=a.returned_within_60_days, date=a.date)
    print(explain(res))


if __name__ == "__main__":
    main()
