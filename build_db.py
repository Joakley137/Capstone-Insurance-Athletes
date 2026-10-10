"""
build_db.py — build the soccer injury MySQL database the models read.

    mysql -u root -e "CREATE DATABASE sports_injury CHARACTER SET utf8mb4"
    python build_db.py data/football-datasets [--db mysql+pymysql://root@localhost/sports_injury]

Drops and recreates the tables, then loads the Transfermarkt CSVs from salimt/football-datasets
(player_injuries*, player_profiles, player_performances, found anywhere under the folder).
Connects to --db, else $SPORTS_DB_URL, else mysql+pymysql://root@localhost/sports_injury.

Tables: injuries, players, player_seasons; view v_injuries_with_age (injuries + age and position).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import sqlalchemy as sa

DB_URL = os.environ.get("SPORTS_DB_URL", "mysql+pymysql://root@localhost/sports_injury")

SCHEMA = """
DROP VIEW IF EXISTS v_injuries_with_age;
DROP TABLE IF EXISTS injuries, players, player_seasons, sources, load_log;

-- One row per injury event.
CREATE TABLE injuries (
    injury_id     INT PRIMARY KEY AUTO_INCREMENT,
    ext_player_id VARCHAR(20),     -- Transfermarkt player id
    injury_date   DATE,
    return_date   DATE,            -- NULL while still out
    days_missed   INT,
    body_part     VARCHAR(50),     -- parsed from injury_desc
    injury_desc   TEXT,
    INDEX ix_inj_player (ext_player_id)
);

-- One row per player.
CREATE TABLE players (
    ext_player_id  VARCHAR(20) PRIMARY KEY,
    date_of_birth  DATE,
    position_group VARCHAR(50),    -- Goalkeeper | Defender | Midfield | Attack
    date_of_death  DATE
);

-- Minutes and appearances per player and season, summed over competitions.
CREATE TABLE player_seasons (
    ext_player_id VARCHAR(20),
    season_start  DATE,            -- '23/24' -> 2023-07-01 to 2024-06-30, '2023' -> calendar year
    season_end    DATE,
    minutes       DOUBLE,
    appearances   INT,
    PRIMARY KEY (ext_player_id, season_start, season_end)
);

CREATE VIEW v_injuries_with_age AS
SELECT i.*,
       ROUND(DATEDIFF(i.injury_date, p.date_of_birth) / 365.25, 2) AS age_at_injury,
       p.position_group
FROM injuries i
LEFT JOIN players p ON p.ext_player_id = i.ext_player_id;
"""

BODY_PARTS = [
    ("achilles", "Achilles"), ("acl", "Knee"), ("mcl", "Knee"), ("meniscus", "Knee"), ("patell", "Knee"),
    ("knee", "Knee"), ("cruciate", "Knee"), ("inner ligament", "Knee"), ("outer ligament", "Knee"),
    ("syndesmo", "Ankle"), ("ankle", "Ankle"), ("hamstring", "Hamstring"), ("groin", "Groin"), ("adductor", "Groin"),
    ("pubic", "Groin"), ("pubalgia", "Groin"),
    ("calf", "Calf"), ("quad", "Quadriceps"), ("thigh", "Thigh"), ("dead leg", "Thigh"), ("hip", "Hip"),
    ("fibula", "Lower leg"), ("tibia", "Lower leg"), ("shin", "Lower leg"), ("back", "Back"),
    ("spine", "Back"), ("lumbar", "Back"), ("shoulder", "Shoulder"), ("rotator", "Shoulder"), ("labrum", "Shoulder"),
    ("collarbone", "Shoulder"), ("clavicle", "Shoulder"),
    ("elbow", "Elbow"), ("ucl", "Elbow"), ("tommy john", "Elbow"), ("wrist", "Wrist"), ("hand", "Hand"),
    ("finger", "Hand"), ("thumb", "Hand"), ("foot", "Foot"), ("toe", "Foot"), ("plantar", "Foot"),
    ("metatars", "Foot"), ("concussion", "Head"), ("head", "Head"), ("nose", "Head"), ("face", "Head"),
    ("facial", "Head"), ("cheek", "Head"), ("jaw", "Head"), ("eye", "Head"), ("skull", "Head"),
    ("neck", "Neck"), ("rib", "Ribs"), ("oblique", "Core"),
    ("abdomin", "Core"), ("chest", "Chest"), ("pectoral", "Chest"), ("illness", "Illness"), ("covid", "Illness"),
    ("corona", "Illness"), ("ill", "Illness"), ("flu", "Illness"), ("influenza", "Illness"), ("cold", "Illness"),
    ("fever", "Illness"), ("virus", "Illness"), ("infection", "Illness"), ("stomach", "Illness"),
    ("quarantine", "Illness"),
    # last, so a named muscle (hamstring, calf, ...) wins over the generic word
    ("muscle", "Muscle (unspecified)"), ("muscular", "Muscle (unspecified)"),
]


def body_part(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for kw, part in BODY_PARTS:
        if re.search(r"\b" + re.escape(kw), t):
            return part
    return None


def iso(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def season_dates(season):
    """'23/24' -> 2023-07-01 .. 2024-06-30;  '2023' -> 2023-01-01 .. 2023-12-31."""
    s = season.astype(str).str.strip()
    yy = pd.to_numeric(s.str.extract(r"^(\d{2})/\d{2}$")[0], errors="coerce").astype(float)
    split_year = yy + np.where(yy > 50, 1900, 2000)
    cal_year = pd.to_numeric(s.str.extract(r"^(\d{4})$")[0], errors="coerce").astype(float)
    start = pd.to_datetime(dict(year=split_year.fillna(cal_year), month=np.where(yy.notna(), 7, 1), day=1),
                           errors="coerce")
    end = pd.to_datetime(dict(year=split_year.fillna(cal_year - 1) + 1, month=np.where(yy.notna(), 6, 12),
                              day=np.where(yy.notna(), 30, 31)), errors="coerce")
    return start.dt.strftime("%Y-%m-%d"), end.dt.strftime("%Y-%m-%d")


def injuries(path):
    raw = pd.read_csv(path, low_memory=False)
    return pd.DataFrame({
        "ext_player_id": raw["player_id"].astype(str),
        "injury_date": iso(raw["from_date"]),
        "return_date": iso(raw["end_date"]),
        "days_missed": pd.to_numeric(raw["days_missed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"),
        "body_part": raw["injury_reason"].map(body_part),
        "injury_desc": raw["injury_reason"],
    })


def players(path):
    raw = pd.read_csv(path, low_memory=False).drop_duplicates("player_id")
    return pd.DataFrame({
        "ext_player_id": raw["player_id"].astype(str),
        "date_of_birth": iso(raw["date_of_birth"]),
        "position_group": raw["main_position"],
        "date_of_death": iso(raw["date_of_death"]),
    })


def player_seasons(path):
    """Its `minutes_played` column is really minutes PER GOAL (blank when the player didn't score), so
    minutes = minutes_played x goals where there are goals. Otherwise minutes are estimated from
    appearances: 90 per full game, 73.5 when subbed off, 18.5 when subbed on. Fitted on the seasons where
    minutes are known, that estimate has R^2 = 0.996 and a median error of 3%."""
    raw = pd.read_csv(path, low_memory=False)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").fillna(0)
    apps, sub_in, sub_out, goals = num("nb_on_pitch"), num("subed_in"), num("subed_out"), num("goals")
    exact = pd.to_numeric(raw["minutes_played"], errors="coerce") * goals
    estimate = 90 * (apps - sub_in - sub_out).clip(lower=0) + 73.5 * sub_out + 18.5 * sub_in
    start, end = season_dates(raw["season_name"])
    out = pd.DataFrame({
        "ext_player_id": raw["player_id"].astype(str), "season_start": start, "season_end": end,
        "minutes": exact.where((goals > 0) & exact.notna(), estimate).round(), "appearances": apps,
    })
    return out.groupby(["ext_player_id", "season_start", "season_end"], as_index=False).sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", help="folder holding salimt/football-datasets")
    p.add_argument("--db", default=DB_URL, help="SQLAlchemy database URL")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.repo, "**", "*.csv"), recursive=True)
    find = lambda test: next((f for f in sorted(files) if test(os.path.basename(f).lower())), None)
    paths = {"injuries": find(lambda n: n.startswith("player_injuries")),
             "players": find(lambda n: n.startswith("player_profiles")),
             "player_seasons": find(lambda n: n.startswith("player_performances"))}
    if not all(paths.values()):
        sys.exit(f"Missing CSVs under {args.repo}: {paths} (player_performances is a Git LFS file: "
                 "run `git lfs pull` there)")

    engine = sa.create_engine(args.db)
    with engine.begin() as con:
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                con.exec_driver_sql(stmt)
        for table, load in [("injuries", injuries), ("players", players), ("player_seasons", player_seasons)]:
            df = load(paths[table])
            df.to_sql(table, con, if_exists="append", index=False, chunksize=10_000)
            print(f"  {table}: {len(df):,} rows from {paths[table]}")


if __name__ == "__main__":
    main()
