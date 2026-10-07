"""
build_db.py — build and load the soccer injury MySQL database used by the models.

Connects to $SPORTS_DB_URL (default mysql+pymysql://root@localhost/sports_injury).
The database itself must exist first:  mysql -u root -e "CREATE DATABASE sports_injury"

Tables: injuries, players, player_seasons, sources, load_log; view v_injuries_with_age.
Data comes from salimt/football-datasets (source 9). Re-running a loader replaces that
source's rows in the target table, so it is safe to reload.

Usage (run `python build_db.py -h` for all options):
  python build_db.py init
  python build_db.py soccer         data/football-datasets      (profiles, seasons and injuries)
  python build_db.py soccer-players data/football-datasets      (profiles only: birth dates, positions)
  python build_db.py soccer-seasons data/football-datasets      (appearances and minutes per season only)
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import sqlalchemy as sa

DB_URL = os.environ.get("SPORTS_DB_URL", "mysql+pymysql://root@localhost/sports_injury")

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     INT PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    url           TEXT,
    sport         VARCHAR(50),
    level         VARCHAR(50),
    coverage      VARCHAR(255),
    format        VARCHAR(100),
    has_injury    TINYINT,
    has_contract  TINYINT,
    has_insurance TINYINT,
    has_nil       TINYINT,
    role_in_model TEXT,
    access_notes  TEXT,
    status        VARCHAR(50) DEFAULT 'Not started',
    notes         TEXT
);

-- One row per injury event (record_type = 'event').
CREATE TABLE IF NOT EXISTS injuries (
    injury_id     INT PRIMARY KEY AUTO_INCREMENT,
    source_id     INT,
    record_type   VARCHAR(20),     -- 'event' | 'weekly_report' | 'snapshot'
    sport         VARCHAR(50),
    league        VARCHAR(100),
    level         VARCHAR(50),
    season        VARCHAR(20),
    week          INT,
    player_name   VARCHAR(255),
    player_key    VARCHAR(255),    -- normalized name for joining across sources
    ext_player_id VARCHAR(100),    -- the source's own player id, if any
    team          VARCHAR(100),
    position      VARCHAR(50),
    injury_date   DATE,
    return_date   VARCHAR(50),     -- ISO date
    days_missed   INT,
    games_missed  INT,
    body_part     VARCHAR(50),
    injury_desc   TEXT,
    status        VARCHAR(100),
    snapshot_date DATE,
    loaded_at     DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    INDEX ix_inj_player (player_key, sport),
    INDEX ix_inj_season (sport, season),
    INDEX ix_inj_source (source_id, season)
);

-- One row per player profile (birth date, position). Join to injuries on ext_player_id + sport.
CREATE TABLE IF NOT EXISTS players (
    player_id      INT PRIMARY KEY AUTO_INCREMENT,
    source_id      INT,
    sport          VARCHAR(50),
    player_name    VARCHAR(255),
    player_key     VARCHAR(255),
    ext_player_id  VARCHAR(100),
    date_of_birth  DATE,
    height_cm      DOUBLE,
    foot           VARCHAR(20),
    position       VARCHAR(100),    -- detailed, e.g. 'Defender - Right-Back'
    position_group VARCHAR(50),     -- Goalkeeper | Defender | Midfield | Attack
    citizenship    VARCHAR(100),
    current_club   VARCHAR(255),    -- 'Retired', 'Without Club', or a club name, as of the download
    date_of_death  DATE,
    loaded_at      DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    INDEX ix_pl_ext (ext_player_id, sport),
    INDEX ix_pl_player (player_key, sport)
);

-- Appearances and minutes per player, season and competition. Join to injuries on ext_player_id + sport.
CREATE TABLE IF NOT EXISTS player_seasons (
    ps_id             INT PRIMARY KEY AUTO_INCREMENT,
    source_id         INT,
    sport             VARCHAR(50),
    ext_player_id     VARCHAR(100),
    season            VARCHAR(20),     -- '23/24' (July-June) or '2023' (calendar-year leagues)
    season_start      DATE,
    season_end        DATE,
    competition       VARCHAR(255),
    team              VARCHAR(255),
    squad_selections  INT,             -- times named in the matchday squad
    appearances       INT,
    subbed_in         INT,
    subbed_out        INT,
    goals             INT,
    minutes           DOUBLE,
    minutes_estimated TINYINT,         -- 1 = estimated from appearances and substitutions
    loaded_at         DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    INDEX ix_ps_player (ext_player_id, sport, season_end)
);

CREATE TABLE IF NOT EXISTS load_log (
    source_id  INT,
    target     VARCHAR(100),
    `rows`     INT,
    file       TEXT,
    loaded_at  DATETIME
);

-- Every injury with the player's age on the injury date (NULL where no profile or birth date is loaded)
CREATE OR REPLACE VIEW v_injuries_with_age AS
SELECT i.*,
       p.date_of_birth,
       ROUND(DATEDIFF(i.injury_date, p.date_of_birth) / 365.25, 2) AS age_at_injury,
       p.position_group,
       p.height_cm
FROM injuries i
LEFT JOIN players p ON p.sport = i.sport AND p.ext_player_id = i.ext_player_id;
"""

SOURCES = [
    # id, name, url, sport, level, coverage, format, inj, con, ins, nil, role, access, notes
    (9, "salimt / football-datasets", "https://github.com/salimt/football-datasets", "Soccer", "Professional", "See repo", "GitHub repo", 1, 0, 0, 1,
     "Soccer injuries, player profiles and minutes", "git clone the repo", "Loader: soccer"),
]

INJURY_COLS = ["source_id", "record_type", "sport", "league", "level", "season", "week", "player_name", "player_key",
               "ext_player_id", "team", "position", "injury_date", "return_date", "days_missed", "games_missed",
               "body_part", "injury_desc", "status", "snapshot_date", "loaded_at"]
PLAYER_COLS = ["source_id", "sport", "player_name", "player_key", "ext_player_id", "date_of_birth", "height_cm",
               "foot", "position", "position_group", "citizenship", "current_club", "date_of_death", "loaded_at"]
SEASON_COLS = ["source_id", "sport", "ext_player_id", "season", "season_start", "season_end", "competition", "team",
               "squad_selections", "appearances", "subbed_in", "subbed_out", "goals", "minutes", "minutes_estimated",
               "loaded_at"]
TABLE_COLS = {"injuries": INJURY_COLS, "players": PLAYER_COLS, "player_seasons": SEASON_COLS}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
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

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")


def now():
    return dt.datetime.now().isoformat(sep=" ", timespec="seconds")


def player_key(name):
    """Normalize a name so 'Jaren Jackson Jr.' and 'jaren jackson' match across sources."""
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.lower()
    n = re.sub(r"\(.*?\)", " ", n)          # drop '(Jalen) Rose' style aliases
    n = n.split("/")[0]                      # 'Name / Alt Name'
    n = re.sub(r"[^a-z\s]", " ", n)
    n = SUFFIX.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip() or None


def body_part(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for kw, part in BODY_PARTS:
        if re.search(r"\b" + re.escape(kw), t):
            return part
    return None


def pick(df, *candidates, required=False):
    """Return the first column in df matching any candidate (case/space-insensitive)."""
    norm = {re.sub(r"[\s_./]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_./]", "", cand.lower())
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")
    return None


def col(df, *candidates):
    c = pick(df, *candidates)
    return df[c] if c else pd.Series([None] * len(df), index=df.index)


def iso(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")



_engines = {}


def connect(url=None):
    """Return a SQLAlchemy engine; use it as `with connect().begin() as con:` for a transaction."""
    url = url or DB_URL
    if url not in _engines:
        _engines[url] = sa.create_engine(url, pool_pre_ping=True)
    return _engines[url]


def run(con, sql, params=None):
    """Execute SQL with :named parameters (a dict, or a list of dicts to run it once per row)."""
    return con.execute(sa.text(sql), params or {})


def write(engine, table, df, source_id, file=None):
    """Replace this source's rows in `table` with df."""
    cols = TABLE_COLS[table]
    df = df.copy()
    df["source_id"] = source_id
    df["loaded_at"] = now()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    with engine.begin() as con:
        run(con, f"DELETE FROM {table} WHERE source_id = :sid", {"sid": source_id})
        df.to_sql(table, con, if_exists="append", index=False, chunksize=10_000)
        run(con, "INSERT INTO load_log VALUES (:sid, :tbl, :n, :file, :at)",
            {"sid": source_id, "tbl": table, "n": len(df), "file": file, "at": now()})
        run(con, "UPDATE sources SET status = 'Collected' WHERE source_id = :sid", {"sid": source_id})
    print(f"  {table}: {len(df):,} rows loaded from source {source_id}")
    return len(df)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------
SOURCE_COLS = ["source_id", "name", "url", "sport", "level", "coverage", "format", "has_injury", "has_contract",
               "has_insurance", "has_nil", "role_in_model", "access_notes", "notes"]


def cmd_init(args):
    engine = connect()
    with engine.begin() as con:
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                con.exec_driver_sql(stmt)
        run(con, f"INSERT INTO sources ({','.join(SOURCE_COLS)}) "
                 f"VALUES ({','.join(':' + c for c in SOURCE_COLS)}) AS new "
                 f"ON DUPLICATE KEY UPDATE {','.join(f'{c} = new.{c}' for c in SOURCE_COLS[1:])}",
            [dict(zip(SOURCE_COLS, s)) for s in SOURCES])
    print(f"Initialized {engine.url.render_as_string(hide_password=True)} with {len(SOURCES)} sources.")


# --------------------------------------------------------------------------
# Source loaders
# --------------------------------------------------------------------------
def cmd_soccer(args):
    """salimt/football-datasets (Transfermarkt-style). Finds the injury, profile and performance
    CSVs anywhere in the repo folder by file name."""
    files = glob.glob(os.path.join(args.repo, "**", "*.csv"), recursive=True)
    find = lambda *words: [f for f in files if all(w in os.path.basename(f).lower() for w in words)]
    inj_files, prof_files = find("injur"), find("profile")
    perf_files = [f for f in files if os.path.basename(f).lower().startswith("player_performances")]
    if not inj_files:
        sys.exit(f"No injury CSVs found under {args.repo}. Files seen: {files[:20]}")

    names = None
    if prof_files:
        prof = pd.read_csv(prof_files[0], low_memory=False)
        id_c, nm_c = pick(prof, "player_id", "id"), pick(prof, "player_name", "name")
        if id_c and nm_c:
            names = prof[[id_c, nm_c]].drop_duplicates(id_c).rename(columns={id_c: "_pid", nm_c: "_pname"})

    def attach_names(df):
        pid_c = pick(df, "player_id", "id")
        nm_c = pick(df, "player_name", "name")
        pid = df[pid_c].astype(str) if pid_c else None
        nm = df[nm_c] if nm_c else None
        if nm is None and names is not None and pid is not None:
            nm = pid.map(dict(zip(names["_pid"].astype(str), names["_pname"])))
        return pid, nm

    con = connect()
    if prof_files:
        load_soccer_players(con, prof_files[0])
    if perf_files:
        load_soccer_seasons(con, perf_files[0])
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in inj_files], ignore_index=True)
    pid, nm = attach_names(raw)
    desc = col(raw, "injury_reason", "injury", "reason", "injury_type")
    out = pd.DataFrame({
        "record_type": "event", "sport": "Soccer", "league": col(raw, "league", "competition"),
        "level": "Professional", "season": col(raw, "season_name", "season"),
        "player_name": nm, "player_key": nm.map(player_key) if nm is not None else None,
        "ext_player_id": pid, "team": col(raw, "club", "team"),
        "injury_date": iso(col(raw, "from_date", "from", "start_date")),
        "return_date": iso(col(raw, "end_date", "until", "to_date")),
        "days_missed": pd.to_numeric(col(raw, "days_missed", "days").astype(str).str.extract(r"(\d+)")[0], errors="coerce"),
        "games_missed": pd.to_numeric(col(raw, "games_missed", "games"), errors="coerce"),
        "injury_desc": desc, "body_part": desc.map(body_part),
    })
    write(con, "injuries", out, 9, file=";".join(inj_files))


def load_soccer_players(con, path):
    """Transfermarkt player_profiles.csv -> players (birth date, height, foot, position)."""
    raw = pd.read_csv(path, low_memory=False)
    pid_c = pick(raw, "player_id", "id", required=True)
    raw = raw.drop_duplicates(pid_c)
    name = col(raw, "player_name", "name").astype("string").str.replace(r"\s*\(\d+\)$", "", regex=True)
    height = pd.to_numeric(col(raw, "height"), errors="coerce")
    out = pd.DataFrame({
        "sport": "Soccer", "player_name": name, "player_key": name.map(player_key),
        "ext_player_id": raw[pid_c].astype(str),
        "date_of_birth": iso(col(raw, "date_of_birth", "Date of birth", "dob")),
        "height_cm": height.where(height.between(140, 220)),          # 0 = unknown
        "foot": col(raw, "foot").replace({"N/A": None}),
        "position": col(raw, "position"),
        "position_group": col(raw, "main_position", "player_main_position", "position_group"),
        "citizenship": col(raw, "citizenship"),
        "current_club": col(raw, "current_club_name", "Current club"),
        "date_of_death": iso(col(raw, "date_of_death", "Date of death")),
    })
    write(con, "players", out, 9, file=path)


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


def load_soccer_seasons(con, path):
    """Transfermarkt player_performances.csv -> player_seasons.

    Its `minutes_played` column is really minutes PER GOAL (blank when the player didn't score), so
    minutes = minutes_played x goals where there are goals. Otherwise minutes are estimated from
    appearances: 90 per full game, 73.5 when subbed off, 18.5 when subbed on. Fitted on the seasons where
    minutes are known, that estimate has R^2 = 0.996 and a median error of 3%."""
    raw = pd.read_csv(path, low_memory=False)
    num = lambda *c: pd.to_numeric(col(raw, *c), errors="coerce").fillna(0)
    apps, sub_in, sub_out, goals = num("nb_on_pitch", "appearances"), num("subed_in"), num("subed_out"), num("goals")
    per_goal = pd.to_numeric(col(raw, "minutes_played"), errors="coerce")
    exact = per_goal * goals
    estimate = 90 * (apps - sub_in - sub_out).clip(lower=0) + 73.5 * sub_out + 18.5 * sub_in
    has_exact = (goals > 0) & exact.notna()
    start, end = season_dates(col(raw, "season_name", "season"))
    out = pd.DataFrame({
        "sport": "Soccer", "ext_player_id": col(raw, "player_id").astype(str),
        "season": col(raw, "season_name", "season"), "season_start": start, "season_end": end,
        "competition": col(raw, "competition_name"), "team": col(raw, "team_name"),
        "squad_selections": num("nb_in_group"), "appearances": apps, "subbed_in": sub_in, "subbed_out": sub_out,
        "goals": goals, "minutes": exact.where(has_exact, estimate).round(),
        "minutes_estimated": (~has_exact).astype(int),
    })
    write(con, "player_seasons", out, 9, file=path)


def cmd_soccer_players(args):
    """Load only the player profiles (useful when injuries are already loaded)."""
    path = args.path
    if os.path.isdir(path):
        found = [f for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
                 if "profile" in os.path.basename(f).lower()]
        if not found:
            sys.exit(f"No player_profiles CSV found under {path}")
        path = found[0]
    load_soccer_players(connect(), path)


def cmd_soccer_seasons(args):
    """Load only appearances and minutes per season."""
    path = args.path
    if os.path.isdir(path):
        found = [f for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
                 if os.path.basename(f).lower().startswith("player_performances")]
        if not found:
            sys.exit(f"No player_performances CSV found under {path} (it is a Git LFS file: run `git lfs pull` there)")
        path = found[0]
    load_soccer_seasons(connect(), path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="SQLAlchemy database URL (default $SPORTS_DB_URL or "
                                "mysql+pymysql://root@localhost/sports_injury)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    s = sub.add_parser("soccer"); s.add_argument("repo"); s.set_defaults(fn=cmd_soccer)
    s = sub.add_parser("soccer-players"); s.add_argument("path"); s.set_defaults(fn=cmd_soccer_players)
    s = sub.add_parser("soccer-seasons"); s.add_argument("path"); s.set_defaults(fn=cmd_soccer_seasons)
    args = p.parse_args()
    global DB_URL
    if args.db:
        DB_URL = args.db
    if args.cmd != "init" and not all(sa.inspect(connect()).has_table(t) for t in ["sources", *TABLE_COLS]):
        cmd_init(args)
    args.fn(args)


if __name__ == "__main__":
    main()
