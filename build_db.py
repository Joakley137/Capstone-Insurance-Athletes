"""
build_db.py — build and load the sports injury / contract MySQL database.

Connects to $SPORTS_DB_URL (default mysql+pymysql://root@localhost/sports_injury).
The database itself must exist first:  mysql -u root -e "CREATE DATABASE sports_injury"

Every loader writes two things:
  1. a raw_<name> table with the source file exactly as downloaded (nothing lost), and
  2. normalized rows in the shared tables (injuries, contracts, market_values),
     so every sport can be queried the same way.

Re-running a loader replaces that source's rows, so it is safe to reload.

Usage (run `python build_db.py -h` for all options):
  python build_db.py init
  python build_db.py nba-injuries  data/nba_injuries/injuries_2010-2020.csv
  python build_db.py nba-salaries  data/nba_salaries/<file>.csv
  python build_db.py nfl           --seasons 2012-2025
  python build_db.py soccer        data/football-datasets
  python build_db.py mlb-fangraphs data/fangraphs_2023.csv --season 2023
  python build_db.py espn-nba
  python build_db.py import-csv    data/ncaa_isp.csv --source-id 3 --table injuries \
                                   --sport Multi-sport --level College \
                                   --map "Injury Date=injury_date,Sport=league,Body Part=body_part"
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys

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

-- One row per injury event, weekly report line, or live snapshot line (see record_type).
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
    return_date   VARCHAR(50),     -- ISO date, except ESPN snapshots which give free text like 'Oct 22'
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

-- Amounts are always stored in full US dollars (not millions).
CREATE TABLE IF NOT EXISTS contracts (
    contract_id   INT PRIMARY KEY AUTO_INCREMENT,
    source_id     INT,
    sport         VARCHAR(50),
    league        VARCHAR(100),
    level         VARCHAR(50),
    player_name   VARCHAR(255),
    player_key    VARCHAR(255),
    ext_player_id VARCHAR(100),
    team          VARCHAR(100),
    position      VARCHAR(50),
    season        VARCHAR(20),     -- for per-season salary rows
    year_signed   INT,             -- for whole-contract rows
    years         INT,
    total_value   DOUBLE,
    apy           DOUBLE,
    guaranteed    DOUBLE,
    salary        DOUBLE,
    loaded_at     DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    INDEX ix_con_player (player_key, sport)
);

CREATE TABLE IF NOT EXISTS market_values (
    mv_id         INT PRIMARY KEY AUTO_INCREMENT,
    source_id     INT,
    sport         VARCHAR(50),
    league        VARCHAR(100),
    player_name   VARCHAR(255),
    player_key    VARCHAR(255),
    ext_player_id VARCHAR(100),
    value_date    DATE,
    market_value  DOUBLE,
    currency      VARCHAR(10),
    loaded_at     DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    INDEX ix_mv_player (player_key, sport)
);

CREATE TABLE IF NOT EXISTS insurance_cases (
    case_id            INT PRIMARY KEY AUTO_INCREMENT,
    player_name        VARCHAR(255),
    player_key         VARCHAR(255),
    sport              VARCHAR(50),
    team               VARCHAR(100),
    contract_total     DOUBLE,
    guaranteed         DOUBLE,
    injury             TEXT,
    seasons_affected   VARCHAR(50),
    games_missed       INT,
    reported_payout    DOUBLE,
    source_ids         VARCHAR(100),
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS load_log (
    source_id  INT,
    target     VARCHAR(100),
    `rows`     INT,
    file       TEXT,
    loaded_at  DATETIME
);

-- Injury totals per player per season (event + snapshot rows, with weekly reports counted as report-weeks)
CREATE OR REPLACE VIEW v_injury_summary AS
SELECT sport, league, level, player_key,
       MAX(player_name)                                        AS player_name,
       season,
       SUM(CASE WHEN record_type = 'event' THEN 1 ELSE 0 END)  AS injury_events,
       SUM(CASE WHEN record_type = 'weekly_report'
                 AND status IN ('Out','Doubtful','IR') THEN 1 ELSE 0 END) AS weeks_out,
       SUM(COALESCE(days_missed, 0))                           AS days_missed,
       SUM(COALESCE(games_missed, 0))                          AS games_missed
FROM injuries
GROUP BY sport, league, level, player_key, season;

-- Career injury totals per player, joined to every contract/salary row for that player
CREATE OR REPLACE VIEW v_contracts_with_injuries AS
SELECT c.*,
       i.career_injury_events,
       i.career_days_missed,
       i.career_weeks_out
FROM contracts c
LEFT JOIN (
    SELECT sport, player_key,
           SUM(injury_events) AS career_injury_events,
           SUM(days_missed)   AS career_days_missed,
           SUM(weeks_out)     AS career_weeks_out
    FROM v_injury_summary GROUP BY sport, player_key
) i ON i.sport = c.sport AND i.player_key = c.player_key;
"""

SOURCES = [
    # id, name, url, sport, level, coverage, format, inj, con, ins, nil, role, access, notes
    (1, "ESPN NBA Injuries", "https://www.espn.com/nba/injuries", "NBA", "Professional", "2026-27 season (live)", "Live web page", 1, 0, 0, 0,
     "Current injury status by team", "Scraped by `espn-nba`; take dated snapshots", "Updates daily"),
    (2, "Covers NBA Injury Report", "https://www.covers.com/sport/basketball/nba/injuries", "NBA", "Professional", "2025-26 season", "Live web page", 1, 0, 0, 0,
     "Cross-check ESPN", "No loader yet; export/copy to CSV and use import-csv", ""),
    (3, "NCAA Injury Surveillance Program - Data Requests", "https://ncaaorg.sidearmsports.com/sports/2018/10/3/ncaa-injury-surveillance-program-data-requests.aspx", "Multi-sport", "College", "Varies by request", "Data request", 1, 0, 0, 0,
     "College injury base rates", "Formal request required; load with import-csv once received", "Start request early"),
    (4, "ESPN UK - NFL insurance policies on star players", "https://www.espn.co.uk/nfl/story/_/id/41274295/nfl-insurance-policies-star-players-aaron-rodgers-tua-tagovailoa-jared-goff-joe-burrow-christian-mccaffrey", "NFL", "Professional", "~2024 (article)", "Article", 0, 1, 1, 0,
     "Payout calibration", "Manual entry into insurance_cases", "Rodgers, Tagovailoa, Goff, Burrow, McCaffrey; details unverified"),
    (5, "Kaggle - NBA Injuries 2010-2020", "https://www.kaggle.com/datasets/ghopkins/nba-injuries-2010-2018", "NBA", "Professional", "2010-2020", "Downloadable dataset", 1, 0, 0, 0,
     "Historical NBA injury events", "kaggle datasets download -d ghopkins/nba-injuries-2010-2018", "Loader: nba-injuries"),
    (6, "Marca - Olympic hockey insurance", "https://www.marca.com/en/nhl/2026/02/10/698b16a7e2704e75698b45bc.html", "NHL", "Olympic / International", "Feb 2026 (article)", "Article", 0, 0, 1, 0,
     "Case study", "Manual entry into insurance_cases", "Details unverified"),
    (7, "ESPN - MLB (2001)", "https://www.espn.com/mlb/s/2001/0307/1136588.html", "MLB", "Professional", "2001 (article)", "Article", None, None, None, None,
     "Historical MLB reference", "Manual", "Topic not yet classified"),
    (8, "nflreadpy (nflverse)", "https://nflreadpy.nflverse.com/", "NFL", "Professional", "Multi-season", "Python package", 1, 1, 0, 0,
     "NFL injury reports + contracts", "pip install nflreadpy", "Loader: nfl"),
    (9, "salimt / football-datasets", "https://github.com/salimt/football-datasets", "Soccer", "Professional", "See repo", "GitHub repo", 1, 0, 0, 1,
     "Soccer injuries + market values", "git clone the repo", "Loader: soccer"),
    (10, "FanGraphs RosterResource Injury Report", "https://www.fangraphs.com/roster-resource/injury-report?groupby=team&timeframe=all&season=2020", "MLB", "Professional", "2020-present", "Live web page", 1, 0, 0, 0,
     "MLB injury records by season", "Export CSV per season from the page", "Loader: mlb-fangraphs"),
    (11, "AOSSM - Hidden costs of NIL", "https://www.sportsmed.org/membership/sports-medicine-update/summer-2025/the-hidden-costs-of-nil-how-injuries-and-insurance-impact-athlete-earnings", "Multi-sport", "College", "Summer 2025 (article)", "Article", 0, 0, 1, 1,
     "NIL marketability after injury", "Manual", ""),
    (12, "Kaggle - NBA Salaries 2003-2019", "https://www.kaggle.com/datasets/josejatem/nba-salaries-20032019", "NBA", "Professional", "2003-2019", "Downloadable dataset", 0, 1, 0, 0,
     "NBA salary side", "kaggle datasets download -d josejatem/nba-salaries-20032019", "Loader: nba-salaries"),
    (13, "Dawgs By Nature - Deshaun Watson contract cost", "https://www.dawgsbynature.com/cleveland-browns-news/127681/deshaun-watsons-contract-has-cost-the-browns-a-lot-less-than-previously-known", "NFL", "Professional", "2026 (article)", "Article", 0, 1, 1, 0,
     "Case study", "Manual", "Totals conflict with source 14"),
    (14, "ProFootballTalk - Watson insurance clarification", "https://www.nbcsports.com/nfl/profootballtalk/rumor-mill/news/taking-a-closer-look-at-the-deshaun-watson-insurance-issue", "NFL", "Professional", "Jul 2026 (article)", "Article", 0, 1, 1, 0,
     "Corrected Watson figure", "Manual", "Revised ~$88.8M estimate to ~$25.8M"),
]

INSURANCE_CASES = [
    ("Deshaun Watson", "NFL", "Browns", 230e6, 230e6,
     "Shoulder (2023); torn Achilles (Oct 2024, re-torn Jan 2025)", "2023-2025", None, 25.824e6, "13,14",
     "PFT first totaled ~$88.8M in cap credits from NFLPA records, then revised to ~$25.8M."),
    ("Aaron Rodgers", "NFL", None, None, None, None, None, None, None, "4", "Fill in from ESPN UK article"),
    ("Tua Tagovailoa", "NFL", None, None, None, None, None, None, None, "4", "Fill in from ESPN UK article"),
    ("Jared Goff", "NFL", None, None, None, None, None, None, None, "4", "Fill in from ESPN UK article"),
    ("Joe Burrow", "NFL", None, None, None, None, None, None, None, "4", "Fill in from ESPN UK article"),
    ("Christian McCaffrey", "NFL", None, None, None, None, None, None, None, "4", "Fill in from ESPN UK article"),
]

INJURY_COLS = ["source_id", "record_type", "sport", "league", "level", "season", "week", "player_name", "player_key",
               "ext_player_id", "team", "position", "injury_date", "return_date", "days_missed", "games_missed",
               "body_part", "injury_desc", "status", "snapshot_date", "loaded_at"]
CONTRACT_COLS = ["source_id", "sport", "league", "level", "player_name", "player_key", "ext_player_id", "team", "position",
                 "season", "year_signed", "years", "total_value", "apy", "guaranteed", "salary", "loaded_at"]
MV_COLS = ["source_id", "sport", "league", "player_name", "player_key", "ext_player_id", "value_date",
           "market_value", "currency", "loaded_at"]
TABLE_COLS = {"injuries": INJURY_COLS, "contracts": CONTRACT_COLS, "market_values": MV_COLS}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
BODY_PARTS = [
    ("achilles", "Achilles"), ("acl", "Knee"), ("mcl", "Knee"), ("meniscus", "Knee"), ("patell", "Knee"),
    ("knee", "Knee"), ("ankle", "Ankle"), ("hamstring", "Hamstring"), ("groin", "Groin"), ("adductor", "Groin"),
    ("calf", "Calf"), ("quad", "Quadriceps"), ("thigh", "Thigh"), ("hip", "Hip"), ("back", "Back"),
    ("spine", "Back"), ("lumbar", "Back"), ("shoulder", "Shoulder"), ("rotator", "Shoulder"), ("labrum", "Shoulder"),
    ("elbow", "Elbow"), ("ucl", "Elbow"), ("tommy john", "Elbow"), ("wrist", "Wrist"), ("hand", "Hand"),
    ("finger", "Hand"), ("thumb", "Hand"), ("foot", "Foot"), ("toe", "Foot"), ("plantar", "Foot"),
    ("concussion", "Head"), ("head", "Head"), ("neck", "Neck"), ("rib", "Ribs"), ("oblique", "Core"),
    ("abdomin", "Core"), ("chest", "Chest"), ("pectoral", "Chest"), ("illness", "Illness"), ("covid", "Illness"),
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


def dollars(series):
    """Parse '$1,234,567' / '12.5' etc. If values look like millions, scale to dollars."""
    s = pd.to_numeric(series.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    med = s.dropna().median() if s.notna().any() else None
    if med is not None and med < 1000:      # e.g. OverTheCap stores 45.0 meaning $45M
        s = s * 1e6
    return s


def nba_season(date_str):
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return None
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


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


def write(engine, table, df, source_id, raw=None, raw_name=None, file=None, only=None):
    """Replace this source's rows in `table` with df; optionally store the raw frame too.
    `only` narrows what gets replaced, e.g. {"season": "2023"} keeps the source's other seasons."""
    cols = TABLE_COLS[table]
    df = df.copy()
    df["source_id"] = source_id
    df["loaded_at"] = now()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    only = only or {}
    scope = "".join(f" AND {c} = :{c}" for c in only)
    with engine.begin() as con:
        run(con, f"DELETE FROM {table} WHERE source_id = :sid{scope}", {"sid": source_id, **only})
        df.to_sql(table, con, if_exists="append", index=False, chunksize=10_000)
        if raw is not None and raw_name:
            raw.to_sql(f"raw_{raw_name}", con, if_exists="replace", index=False, chunksize=10_000)
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
CASE_COLS = ["player_name", "player_key", "sport", "team", "contract_total", "guaranteed", "injury",
             "seasons_affected", "games_missed", "reported_payout", "source_ids", "notes"]


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
        if run(con, "SELECT COUNT(*) FROM insurance_cases").scalar() == 0:
            run(con, f"INSERT INTO insurance_cases ({','.join(CASE_COLS)}) "
                     f"VALUES ({','.join(':' + c for c in CASE_COLS)})",
                [dict(zip(CASE_COLS, (c[0], player_key(c[0])) + c[1:])) for c in INSURANCE_CASES])
    print(f"Initialized {engine.url.render_as_string(hide_password=True)} with {len(SOURCES)} sources.")


# --------------------------------------------------------------------------
# Source loaders
# --------------------------------------------------------------------------
def cmd_nba_injuries(args):
    """Kaggle ghopkins (Pro Sports Transactions format): Date, Team, Acquired, Relinquished, Notes.
    A 'Relinquished' row = player placed on injured list; the next 'Acquired' row for the
    same player = activated. Those two are paired into one injury event."""
    raw = pd.read_csv(args.path)
    date_c = pick(raw, "Date", required=True)
    team_c = pick(raw, "Team")
    acq_c = pick(raw, "Acquired", required=True)
    rel_c = pick(raw, "Relinquished", required=True)
    note_c = pick(raw, "Notes", "Note")

    df = raw.copy()
    df["_date"] = pd.to_datetime(df[date_c], errors="coerce")
    clean = lambda s: s.astype(str).str.replace("•", "", regex=False).str.strip().replace({"nan": None, "": None})
    df["_acq"] = clean(df[acq_c])
    df["_rel"] = clean(df[rel_c])
    df = df.sort_values("_date")

    outs = df[df["_rel"].notna()].copy()
    outs["player_key"] = outs["_rel"].map(player_key)
    backs = df[df["_acq"].notna()].copy()
    backs["player_key"] = backs["_acq"].map(player_key)
    backs = backs[["player_key", "_date"]].rename(columns={"_date": "_ret"}).sort_values("_ret")

    outs = outs.dropna(subset=["_date"])
    backs = backs.dropna(subset=["_ret"])
    paired = pd.merge_asof(outs.sort_values("_date"), backs, left_on="_date", right_on="_ret",
                           by="player_key", direction="forward", allow_exact_matches=False)

    out = pd.DataFrame({
        "record_type": "event", "sport": "NBA", "league": "NBA", "level": "Professional",
        "player_name": paired["_rel"], "player_key": paired["player_key"],
        "team": paired[team_c] if team_c else None,
        "injury_date": paired["_date"].dt.strftime("%Y-%m-%d"),
        "return_date": paired["_ret"].dt.strftime("%Y-%m-%d"),
        "days_missed": (paired["_ret"] - paired["_date"]).dt.days,
        "injury_desc": paired[note_c] if note_c else None,
    })
    out["season"] = out["injury_date"].map(nba_season)
    out["body_part"] = out["injury_desc"].map(body_part)
    write(connect(), "injuries", out, 5, raw=raw, raw_name="nba_injuries_kaggle", file=args.path)


def cmd_nba_salaries(args):
    raw = pd.read_csv(args.path)
    name_c = pick(raw, "player", "playerName", "Player", "name", required=True)
    out = pd.DataFrame({
        "sport": "NBA", "league": "NBA", "level": "Professional",
        "player_name": raw[name_c],
        "player_key": raw[name_c].map(player_key),
        "ext_player_id": col(raw, "playerId", "player_id", "id"),
        "team": col(raw, "team", "tm", "teamName"),
        "position": col(raw, "position", "pos"),
        "season": col(raw, "season", "seasonStartYear", "year").astype(str),
        "salary": dollars(col(raw, "salary", "Salary", "amount")),
    })
    write(connect(), "contracts", out, 12, raw=raw, raw_name="nba_salaries_kaggle", file=args.path)


def cmd_nfl(args):
    try:
        import nflreadpy as nfl
    except ImportError:
        sys.exit("nflreadpy is not installed. Run: pip install nflreadpy")
    a, b = (int(x) for x in args.seasons.split("-")) if "-" in args.seasons else (int(args.seasons),) * 2
    seasons = list(range(a, b + 1))
    to_pd = lambda x: x.to_pandas() if hasattr(x, "to_pandas") else x   # nflreadpy returns Polars

    con = connect()
    inj = to_pd(nfl.load_injuries(seasons))
    name = col(inj, "full_name", "player_name")
    desc = col(inj, "report_primary_injury").fillna(col(inj, "practice_primary_injury"))
    injuries = pd.DataFrame({
        "record_type": "weekly_report", "sport": "NFL", "league": "NFL", "level": "Professional",
        "season": pd.to_numeric(col(inj, "season"), errors="coerce").astype("Int64").astype(str),
        "week": col(inj, "week"),
        "player_name": name, "player_key": name.map(player_key),
        "ext_player_id": col(inj, "gsis_id"), "team": col(inj, "team"), "position": col(inj, "position"),
        "injury_desc": desc, "body_part": desc.map(body_part),
        "status": col(inj, "report_status").fillna(col(inj, "practice_status")),
        "injury_date": iso(col(inj, "date_modified")),
    })
    write(con, "injuries", injuries, 8, raw=inj, raw_name="nfl_injuries", file=f"nflreadpy {seasons[0]}-{seasons[-1]}")

    con_df = to_pd(nfl.load_contracts())
    pname = col(con_df, "player", "player_name")
    contracts = pd.DataFrame({
        "sport": "NFL", "league": "NFL", "level": "Professional",
        "player_name": pname, "player_key": pname.map(player_key),
        "ext_player_id": col(con_df, "gsis_id", "otc_id"),
        "team": col(con_df, "team"), "position": col(con_df, "position"),
        "year_signed": col(con_df, "year_signed"), "years": col(con_df, "years"),
        "total_value": dollars(col(con_df, "value")),
        "apy": dollars(col(con_df, "apy")),
        "guaranteed": dollars(col(con_df, "guaranteed")),
    })
    nested = [c for c in con_df.columns if con_df[c].map(lambda v: pd.api.types.is_list_like(v)).any()]
    write(con, "contracts", contracts, 8, raw=con_df.drop(columns=nested),   # SQL columns can't hold nested lists
          raw_name="nfl_contracts", file="nflreadpy load_contracts")


def cmd_soccer(args):
    """salimt/football-datasets (Transfermarkt-style). Finds injury, market-value and profile
    CSVs anywhere in the repo folder by file name."""
    files = glob.glob(os.path.join(args.repo, "**", "*.csv"), recursive=True)
    find = lambda *words: [f for f in files if all(w in os.path.basename(f).lower() for w in words)]
    inj_files, mv_files = find("injur"), find("market", "value")
    prof_files = find("profile")
    if not inj_files and not mv_files:
        sys.exit(f"No injury or market-value CSVs found under {args.repo}. Files seen: {files[:20]}")

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
    if inj_files:
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
        write(con, "injuries", out, 9, raw=raw, raw_name="soccer_injuries", file=";".join(inj_files))
    if mv_files:
        raw = pd.concat([pd.read_csv(f, low_memory=False) for f in mv_files], ignore_index=True)
        pid, nm = attach_names(raw)
        out = pd.DataFrame({
            "sport": "Soccer", "player_name": nm,
            "player_key": nm.map(player_key) if nm is not None else None, "ext_player_id": pid,
            "value_date": iso(col(raw, "date_unix", "date", "value_date")),
            "market_value": pd.to_numeric(col(raw, "value", "market_value", "market_value_in_eur"), errors="coerce"),
            "currency": "EUR",
        })
        write(con, "market_values", out, 9, raw=raw, raw_name="soccer_market_values", file=";".join(mv_files))


def cmd_mlb_fangraphs(args):
    """CSV exported from FanGraphs RosterResource injury report (one file per season)."""
    raw = pd.read_csv(args.path)
    nm = col(raw, "Name", "Player")
    desc = col(raw, "Injury / Surgery", "Injury", "InjurySurgery")
    out = pd.DataFrame({
        "record_type": "event", "sport": "MLB", "league": "MLB", "level": "Professional",
        "season": str(args.season), "player_name": nm, "player_key": nm.map(player_key),
        "team": col(raw, "Team"), "position": col(raw, "Pos", "Position"),
        "injury_date": iso(col(raw, "IL Retro Date", "Injury Date", "Date")),
        "return_date": iso(col(raw, "Return Date", "Actual Return")),
        "injury_desc": desc, "body_part": desc.map(body_part),
        "status": col(raw, "Status", "Current Status"),
    })
    out["days_missed"] = (pd.to_datetime(out["return_date"]) - pd.to_datetime(out["injury_date"])).dt.days
    # FanGraphs is one source across many seasons, so only replace this season's rows
    write(connect(), "injuries", out, 10, raw=raw, raw_name=f"mlb_fangraphs_{args.season}", file=args.path,
          only={"season": str(args.season)})


def cmd_espn_nba(args):
    """Snapshot the live ESPN NBA injury page. Each run appends a dated snapshot, so running it
    regularly builds a history. Page layout can change; if it breaks, check the table classes."""
    import requests
    from bs4 import BeautifulSoup
    from io import StringIO
    html = requests.get("https://www.espn.com/nba/injuries",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    soup = BeautifulSoup(html, "lxml")
    frames = []
    for block in soup.select("div.ResponsiveTable"):
        title = block.select_one(".Table__Title")
        table = block.find("table")
        if table is None:
            continue
        t = pd.read_html(StringIO(str(table)))[0]
        t["TEAM"] = title.get_text(strip=True) if title else None
        frames.append(t)
    if not frames:
        sys.exit("No injury tables found; ESPN may have changed its page layout.")
    raw = pd.concat(frames, ignore_index=True)
    today = dt.date.today().isoformat()
    nm = col(raw, "NAME")
    desc = col(raw, "COMMENT")
    out = pd.DataFrame({
        "record_type": "snapshot", "sport": "NBA", "league": "NBA", "level": "Professional",
        "season": nba_season(today), "player_name": nm, "player_key": nm.map(player_key),
        "team": raw["TEAM"], "position": col(raw, "POS"),
        "return_date": col(raw, "EST. RETURN DATE"), "status": col(raw, "STATUS"),
        "injury_desc": desc, "body_part": desc.map(body_part), "snapshot_date": today,
    })
    write(connect(), "injuries", out, 1, file="espn live", only={"snapshot_date": today})


def cmd_import_csv(args):
    """Load any CSV (NCAA ISP data, Covers export, hand-built files) into a normalized table.
    --map renames source columns to schema columns: "Their Col=our_col,Other=our_col2"."""
    raw = pd.read_csv(args.path)
    mapping = dict(kv.split("=", 1) for kv in args.map.split(",")) if args.map else {}
    df = raw.rename(columns={k.strip(): v.strip() for k, v in mapping.items()})
    allowed = set(TABLE_COLS[args.table])
    df = df[[c for c in df.columns if c in allowed]].copy()
    for k, v in (("sport", args.sport), ("level", args.level), ("record_type", args.record_type)):
        if v and k in allowed and k not in df.columns:
            df[k] = v
    if "player_name" in df.columns:
        df["player_key"] = df["player_name"].map(player_key)
    for d in ("injury_date", "return_date", "value_date"):
        if d in df.columns:
            df[d] = iso(df[d])
    if args.table == "injuries" and "injury_desc" in df.columns and "body_part" not in df.columns:
        df["body_part"] = df["injury_desc"].map(body_part)
    for m in ("total_value", "apy", "guaranteed", "salary"):
        if m in df.columns:
            df[m] = dollars(df[m])
    name = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(args.path))[0]).lower()
    write(connect(), args.table, df, args.source_id, raw=raw, raw_name=name, file=args.path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="SQLAlchemy database URL (default $SPORTS_DB_URL or "
                                "mysql+pymysql://root@localhost/sports_injury)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    s = sub.add_parser("nba-injuries"); s.add_argument("path"); s.set_defaults(fn=cmd_nba_injuries)
    s = sub.add_parser("nba-salaries"); s.add_argument("path"); s.set_defaults(fn=cmd_nba_salaries)
    s = sub.add_parser("nfl"); s.add_argument("--seasons", default="2012-2025"); s.set_defaults(fn=cmd_nfl)
    s = sub.add_parser("soccer"); s.add_argument("repo"); s.set_defaults(fn=cmd_soccer)
    s = sub.add_parser("mlb-fangraphs"); s.add_argument("path"); s.add_argument("--season", required=True)
    s.set_defaults(fn=cmd_mlb_fangraphs)
    sub.add_parser("espn-nba").set_defaults(fn=cmd_espn_nba)
    s = sub.add_parser("import-csv"); s.add_argument("path")
    s.add_argument("--source-id", type=int, required=True)
    s.add_argument("--table", choices=list(TABLE_COLS), required=True)
    s.add_argument("--map", default="")
    s.add_argument("--sport"); s.add_argument("--level"); s.add_argument("--record-type", default="event")
    s.set_defaults(fn=cmd_import_csv)
    args = p.parse_args()
    global DB_URL
    if args.db:
        DB_URL = args.db
    if args.cmd != "init" and not sa.inspect(connect()).has_table("sources"):
        cmd_init(args)
    args.fn(args)


if __name__ == "__main__":
    main()
