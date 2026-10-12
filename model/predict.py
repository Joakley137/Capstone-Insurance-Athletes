"""
predict.py — season quote for a player (chance of injury, of an injury beyond the deferment, of a
career-ending injury), or days out and career-ending chance for one described injury.

    python -m model.predict --age 27 --position Defender --league "Premier League" \\
                      --minutes-last-season 2500 --career-minutes 15000 --prior-injuries 4 [--deferment 60]
    python -m model.predict --injury "Cruciate ligament tear" --age 27 --position Defender \\
                      --minutes-last-12-months 2500
    python -m model.predict --csv runs/example_players.csv [--out quotes.csv]   # or runs/example_injuries.csv
    python -m model.predict --refit

    from model import predict
    predict.quote(27, position="Defender", league="Premier League", minutes_last_season=2500)
    predict.injury("Cruciate ligament tear", age=27, position="Defender")

Season quote (pricing.quote): expected injuries from the negative-binomial frequency GLM, the per-injury
chance of lasting more than the deferment and of ending the career from the severity and career GLMs
averaged over the injury mix, combined with the NB zero probability. One set of inputs: minutes last
season is also the career model's minutes in the last 12 months; league and days out last season only
affect the expected injuries; career minutes and date only the career-ending chance.

Per-injury mode (--injury): days out depend on the injury (its severity tier), position, league, age, prior
injuries, a recent re-injury and the same body part injured before (fit on big-five injuries, 2015/16 on).
Minutes in the last 12 months, career minutes and the date only affect the career-ending chance, not days out.

Omitted inputs are set to typical values and listed under `assumed`.
Models are fitted once on all the data (severity on big-five injuries, see pricing.py) and cached in model/fitted/injury_models.pkl (--refit to redo).
"""
import argparse
import datetime as dt
import os
import pickle

import numpy as np
import pandas as pd

from database import build_db
from model import career
from model import frequency as freq
from model import severity as sev

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitted", "injury_models.pkl")
POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
INJURY_TYPES = sorted({label for _, label in sev.NATURE if label != "Not an injury"} | {"Unspecified"})
BODY_PARTS = sorted({part for _, part in build_db.BODY_PARTS} | {"Unknown"})


# --------------------------------------------------------------------------
# Fitting and saving
# --------------------------------------------------------------------------
def _slim(model):
    """Drop statsmodels' stored training data to keep the pickle small."""
    model.res.remove_data()
    return model


def fit_and_save(path=MODEL_FILE):
    """Fit severity, career and frequency models on all the soccer data (pricing.fit: severity on the
    big-five injuries, career on all leagues) and pickle them."""
    from model import pricing      # imported here: pricing imports this module
    print("Fitting the models on all the soccer data (a few minutes, once) ...", flush=True)
    data, cdata, fdata = sev.build_dataset(), career.build_dataset(), freq.build_dataset()

    # ==================================================================
    # GLM FIT — tiered gamma (days out), logistic (career-ending), negative binomial (injuries per season)
    # ==================================================================
    bundle = pricing.fit(data, cdata, fdata)
    for k in ["tiered", "career", "frequency"]:
        _slim(bundle[k])
    bundle.update(fitted=dt.date.today().isoformat(), n_injuries=bundle["n_insured_injuries"], n_career=len(cdata),
                  n_player_seasons=len(fdata))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved to {path}")
    return bundle


_bundle = None


def models(refit=False):
    """All-data models, fitted once and cached in MODEL_FILE (refit if the cache predates the big-five severity fit)."""
    global _bundle
    if _bundle is None and not refit and os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            _bundle = pickle.load(f)
    if refit or _bundle is None or _bundle.get("population") != "big five":
        _bundle = fit_and_save()
    return _bundle


# --------------------------------------------------------------------------
# Building the feature row
# --------------------------------------------------------------------------
def _rating_class(b, region, nature):
    """Mirror severity.injury_classes: own class, else type pool, region pool, 'Other (rare)'."""
    if (region, nature) in b["class_of"]:
        return b["class_of"][(region, nature)]
    by_type = ("Unspecified injury" if nature == "Unspecified" else nature) + " - other sites"
    for c in [by_type, f"{region} - other types", "Other (rare)"]:
        if c in b["classes"]:
            return c
    return sev.REF_CLASS


def _row(b, injury=None, age=None, position=None, body_part=None, injury_type=None, minutes_last_12_months=None, career_minutes=None, prior_injuries=None, same_body_part_before=False,
         returned_within_60_days=False, date=None, league=None):
    from model import pricing      # imported here: pricing imports this module
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
    if league is None:
        league = b["frequency"].typical["league"]
        assumed.append(f"league = {league}")
    league = pricing.league_code(league)

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
    when = pd.Timestamp(date) if date else pd.Timestamp(dt.date.today())

    region = sev.REGIONS.get(part, part)
    row = {
        "body_part": part, "region": region, "nature": kind,
        "injury_class": _rating_class(b, region, kind), "position_group": position, "league": league,
        "age": float(age), "age_c": float(age) - 26,
        "prior_injuries": float(prior_injuries), "reinjury_60d": int(bool(returned_within_60_days)),
        "prior_same_part": int(bool(same_body_part_before)),
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
    # Don't extrapolate the career model's year trend past its data
    cdf = df.assign(year_c=np.minimum(df["year"], b["career_last_year"]) - 2015)
    for c in ["region", "nature", "position_group"]:
        cdf[c] = cdf[c].astype(str)
    out["career_ending_chance"] = car.predict(cdf).round(4)
    return out


def injury(injury=None, *, age, position=None, body_part=None, injury_type=None, minutes_last_12_months=None, career_minutes=None, prior_injuries=None, same_body_part_before=False,
           returned_within_60_days=False, date=None, league=None):
    """Predict one injury -> pandas Series (`assumed` lists filled-in inputs).

    minutes_last_12_months, career_minutes and date only feed the career-ending
    chance; days out don't use them. league (big five) only feeds days out.
    """
    b = models()
    row, assumed = _row(b, injury, age, position, body_part, injury_type, minutes_last_12_months, career_minutes,
                        prior_injuries, same_body_part_before, returned_within_60_days, date, league)
    out = _predict_frame(b, [row]).iloc[0]
    out["assumed"] = "; ".join(assumed) or "nothing"
    return out


INJURY_COLUMNS = ["injury", "age", "position", "league", "body_part", "injury_type", "minutes_last_12_months",
                  "career_minutes", "prior_injuries", "same_body_part_before", "returned_within_60_days", "date"]
QUOTE_COLUMNS = ["age", "position", "league", "minutes_last_season", "career_minutes", "prior_injuries",
                 "days_out_last_season", "date", "deferment"]


def _csv_args(raw, columns):
    unknown = set(raw.columns) - set(columns) - {"name", "player"}
    if unknown:
        raise ValueError(f"Unknown columns {sorted(unknown)}; expected some of {columns}")
    return [{k: (None if pd.isna(v) else v) for k, v in r.items() if k in columns} for _, r in raw.iterrows()]


def from_csv(path):
    """Per-injury prediction for each CSV row (INJURY_COLUMNS; `name`/`player` passed through)."""
    b = models()
    raw = pd.read_csv(path)
    rows, notes = [], []
    for args in _csv_args(raw, INJURY_COLUMNS):
        for flag in ["same_body_part_before", "returned_within_60_days"]:
            v = args.get(flag)
            args[flag] = str(v).strip().lower() in ("1", "true", "yes", "y") if v is not None else False
        row, assumed = _row(b, **args)
        rows.append(row)
        notes.append("; ".join(assumed) or "nothing")
    out = _predict_frame(b, rows)
    out["assumed"] = notes
    return pd.concat([raw, out], axis=1)


def quote(age, *, position=None, league=None, minutes_last_season=None, career_minutes=None,
          prior_injuries=None, days_out_last_season=None, date=None, deferment=60):
    """Season quote for one player -> pandas Series (see pricing.quote; `assumed` lists filled-in inputs).
    minutes_last_season is also used as minutes in the last 12 months."""
    from model import pricing
    return pricing.quote(age, position, league, minutes_last_season, career_minutes, prior_injuries,
                         days_out_last_season, date, int(deferment or 60), models())


def quotes_from_csv(path):
    """Season quote for each CSV row (QUOTE_COLUMNS; `name`/`player` passed through)."""
    raw = pd.read_csv(path)
    out = pd.DataFrame([quote(**a) for a in _csv_args(raw, QUOTE_COLUMNS)])
    return pd.concat([raw, out.drop(columns="deferment_days")], axis=1)


def _pct(p):
    return "under 0.01%" if p < 0.0001 else f"{p:.2%}" if p < 0.1 else f"{p:.0%}"


def explain(p):
    """Plain-language summary of an injury() result."""
    return "\n".join([
        f"Read as: {p.injury_type} — {p.body_part}  (rating class '{p.rating_class}', "
        f"{p.severity_tier} of {len(models()['tiered'].tier_names)}, higher = longer)",
        f"Expected time out: {p.expected_days:.0f} days",
        f"Likely range: half of similar injuries take under {p.median_days:.0f} days, "
        f"3 in 4 under {p.p75_days:.0f}, 9 in 10 under {p.p90_days:.0f}",
        f"Chance of missing more than 90 days: {p.chance_over_90_days:.0%};  more than 180 days: "
        f"{p.chance_over_180_days:.0%}",
        f"Chance this injury ends the career: {_pct(p.career_ending_chance)}  (average injury: about 0.42%)",
        f"Filled in with typical values: {p.assumed}",
    ])


def explain_quote(s):
    """Plain-language summary of a quote() result."""
    d = int(s.deferment_days)
    return "\n".join([
        f"Expected injuries this season: {s.expected_injuries:.2f}",
        f"Chance of at least one injury: {_pct(s.p_any_injury)}",
        f"Chance of at least one injury lasting more than {d} days (covered): {_pct(s.p_covered_injury)}",
        f"  expected covered injuries: {s.expected_covered_injuries:.3f}; "
        f"a covered injury lasts {s.mean_days_if_covered:.0f} days on average",
        f"Chance of a career-ending injury this season: {_pct(s.p_career_ending)}",
        f"Chance a given injury lasts more than 30 / 60 / 90 / 180 days: "
        + " / ".join(_pct(s[f"q_{x}"]) for x in (30, 60, 90, 180)),
        f"Filled in with typical values: {s.assumed}",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--age", type=float)
    p.add_argument("--position", choices=POSITIONS)
    p.add_argument("--league", help="Premier League, LaLiga, Serie A, Bundesliga, Ligue 1 (or GB1, ES1, IT1, L1, FR1)")
    p.add_argument("--minutes-last-season", "--minutes-last-12-months", dest="minutes", type=float,
                   help="minutes played last season (= last 12 months)")
    p.add_argument("--career-minutes", type=float)
    p.add_argument("--prior-injuries", type=int)
    p.add_argument("--days-out-last-season", type=float, help="season quote only")
    p.add_argument("--deferment", type=int, default=60, help="days before cover starts (season quote)")
    p.add_argument("--date", help="YYYY-MM-DD (default today); career-ending chance only")
    p.add_argument("--injury", help='per-injury mode: a description, e.g. "Hamstring strain", "Broken foot"')
    p.add_argument("--body-part", choices=BODY_PARTS, help="per-injury mode")
    p.add_argument("--injury-type", choices=INJURY_TYPES, help="per-injury mode")
    p.add_argument("--same-body-part-before", action="store_true", help="per-injury mode")
    p.add_argument("--returned-within-60-days", action="store_true",
                   help="per-injury mode: hurt again within 60 days of returning")
    p.add_argument("--csv", help="a row per player (season quotes), or per injury if it has an `injury` column")
    p.add_argument("--out", help="with --csv: write the results here")
    p.add_argument("--refit", action="store_true", help="refit and re-save the models")
    a = p.parse_args()

    if a.refit:
        models(refit=True)
        if not (a.csv or a.age is not None):
            return
    if a.csv:
        per_injury = "injury" in pd.read_csv(a.csv, nrows=0).columns
        res = from_csv(a.csv) if per_injury else quotes_from_csv(a.csv)
        if a.out:
            res.to_csv(a.out, index=False)
            print(f"Wrote {len(res)} rows to {a.out}")
        else:
            pd.set_option("display.width", 250)
            print(res.round(4).to_string(index=False))
        return
    if a.age is None:
        p.error("give --age (season quote; add --injury for one injury), or --csv FILE")
    if a.injury or a.body_part:
        res = injury(a.injury, age=a.age, position=a.position, league=a.league, body_part=a.body_part,
                     injury_type=a.injury_type,
                     minutes_last_12_months=a.minutes, career_minutes=a.career_minutes,
                     prior_injuries=a.prior_injuries, same_body_part_before=a.same_body_part_before,
                     returned_within_60_days=a.returned_within_60_days, date=a.date)
        print(explain(res))
        return
    res = quote(a.age, position=a.position, league=a.league, minutes_last_season=a.minutes,
                career_minutes=a.career_minutes, prior_injuries=a.prior_injuries,
                days_out_last_season=a.days_out_last_season, date=a.date, deferment=a.deferment)
    print(explain_quote(res))


if __name__ == "__main__":
    main()
