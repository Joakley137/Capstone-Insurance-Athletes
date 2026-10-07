"""
sportsdb.py — pull data from the sports_injury MySQL database into pandas.

    import sportsdb as sdb
    inj  = sdb.injuries(sport="Soccer", body_part="knee")
    ppl  = sdb.players(player="messi")
    apps = sdb.player_seasons(sport="Soccer")         # appearances and minutes per season
    ages = sdb.injuries_with_age(sport="Soccer")      # + age_at_injury, position_group
    df   = sdb.query("SELECT body_part, COUNT(*) n FROM injuries GROUP BY 1 ORDER BY n DESC")
    df   = sdb.query("SELECT * FROM injuries WHERE team = :team", {"team": "Arsenal"})

Connects to $SPORTS_DB_URL (default mysql+pymysql://root@localhost/sports_injury).
"""
import os

import pandas as pd
import sqlalchemy as sa

DB_URL = os.environ.get("SPORTS_DB_URL", "mysql+pymysql://root@localhost/sports_injury")
_engines = {}


def connect(url=None):
    url = url or DB_URL
    if url not in _engines:
        _engines[url] = sa.create_engine(url, pool_pre_ping=True)
    return _engines[url]


def query(sql, params=None, url=None):
    """Run any SQL and get a DataFrame back. Parameters are :named, passed as a dict."""
    with connect(url).connect() as con:
        return pd.read_sql_query(sa.text(sql), con, params=params or {})


def _where(**filters):
    """Build a WHERE clause from keyword filters. Lists become IN (...); None is skipped."""
    clauses, params = [], {}

    def bind(v):
        name = f"p{len(params)}"
        params[name] = v
        return f":{name}"

    for key, val in filters.items():
        if val is None:
            continue
        col, _, op = key.partition("__")
        if isinstance(val, (list, tuple, set)):
            clauses.append(f"{col} IN ({','.join(bind(v) for v in val)})")
        elif op == "min":
            clauses.append(f"{col} >= {bind(val)}")
        elif op == "max":
            clauses.append(f"{col} <= {bind(val)}")
        else:
            clauses.append(f"{col} = {bind(val)}")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def sources(**filters):
    w, p = _where(**filters)
    return query(f"SELECT * FROM sources{w} ORDER BY source_id", p)


def injuries(sport=None, league=None, level=None, seasons=None, player=None, body_part=None,
             record_type=None, source_id=None):
    """Normalized injury rows. `player` matches any part of the name (case-insensitive)."""
    w, p = _where(sport=sport, league=league, level=level, season=seasons, body_part=body_part,
                  record_type=record_type, source_id=source_id)
    if player:
        w += (" AND " if w else " WHERE ") + "player_key LIKE :player"
        p["player"] = f"%{player.lower()}%"
    return query(f"SELECT * FROM injuries{w} ORDER BY sport, injury_date", p)


def players(sport=None, player=None):
    """Player profiles: birth date, height, foot, position."""
    w, p = _where(sport=sport)
    if player:
        w += (" AND " if w else " WHERE ") + "player_key LIKE :player"
        p["player"] = f"%{player.lower()}%"
    return query(f"SELECT * FROM players{w}", p)


def player_seasons(sport=None, player_id=None):
    """Appearances and minutes per player, season and competition. player_id is the source's id
    (ext_player_id, the Transfermarkt id for soccer)."""
    w, p = _where(sport=sport, ext_player_id=player_id)
    return query(f"SELECT * FROM player_seasons{w} ORDER BY ext_player_id, season_start", p)


def injuries_with_age(sport=None, seasons=None, body_part=None, record_type=None):
    """Injury rows plus the player's age on the injury date, position group and height."""
    w, p = _where(sport=sport, season=seasons, body_part=body_part, record_type=record_type)
    return query(f"SELECT * FROM v_injuries_with_age{w} ORDER BY sport, injury_date", p)


def body_part_rates(sport=None):
    """How often each body part shows up, and average days missed — rough severity inputs."""
    w, p = _where(sport=sport, record_type="event")
    return query(f"""SELECT sport, body_part, COUNT(*) AS injuries,
                            ROUND(AVG(days_missed), 1) AS avg_days_missed,
                            ROUND(AVG(games_missed), 1) AS avg_games_missed
                     FROM injuries{w}
                     GROUP BY sport, body_part ORDER BY sport, injuries DESC""", p)


def tables():
    """List every table and view, with row counts."""
    names = query("SELECT table_name AS name, "
                  "CASE table_type WHEN 'VIEW' THEN 'view' ELSE 'table' END AS type "
                  "FROM information_schema.tables WHERE table_schema = DATABASE() ORDER BY type, name")
    names["rows"] = [query(f"SELECT COUNT(*) n FROM `{n}`")["n"][0] for n in names["name"]]
    return names


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(tables())
