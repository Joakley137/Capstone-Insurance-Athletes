"""
sportsdb.py — run SQL against the sports_injury MySQL database and get pandas DataFrames back.

    import sportsdb as sdb
    df = sdb.query("SELECT body_part, COUNT(*) n FROM injuries GROUP BY 1 ORDER BY n DESC")
    df = sdb.query("SELECT * FROM players WHERE ext_player_id = :id", {"id": "28003"})

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
