# Sports Injury & Contract Database

A MySQL database (`sports_injury`) plus:

- **build_db.py** downloads nothing on its own; it loads the files you download into the database.
- **sportsdb.py** pulls data back out as pandas DataFrames.
- **explore.ipynb** is a short notebook that connects and pulls a sample of each table.

`python build_db.py init` creates the tables, the 14-source registry and the insurance case studies; the injury and contract tables are empty until you run the loaders.

## Setup

```bash
brew install mysql
brew services start mysql            # runs MySQL in the background, and at login
mysql -u root -e "CREATE DATABASE sports_injury CHARACTER SET utf8mb4"

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_db.py init
```

Both scripts connect to `mysql+pymysql://root@localhost/sports_injury` by default. To use a different server, user or password, set `SPORTS_DB_URL`, e.g. `export SPORTS_DB_URL="mysql+pymysql://user:password@host:3306/sports_injury"`, or pass `--db URL` to `build_db.py`.

**Viewing the data:** any MySQL client works. Free options: [MySQL Workbench](https://dev.mysql.com/downloads/workbench/), [DBeaver Community](https://dbeaver.io/), or the `mysql` command line. Connect to host `localhost`, port `3306`, user `root`, no password.

## Load the data (run from this folder)

| Source | Get it | Load it |
|---|---|---|
| NBA injuries 2010-20 (Kaggle) | `kaggle datasets download -d ghopkins/nba-injuries-2010-2018 -p data/nba_inj --unzip` | `python build_db.py nba-injuries data/nba_inj/<file>.csv` |
| NBA salaries 2003-19 (Kaggle) | `kaggle datasets download -d josejatem/nba-salaries-20032019 -p data/nba_sal --unzip` | `python build_db.py nba-salaries data/nba_sal/<file>.csv` |
| NFL injuries + contracts | (downloaded by nflreadpy) | `python build_db.py nfl --seasons 2012-2025` |
| Soccer (salimt) | `git clone https://github.com/salimt/football-datasets data/football-datasets` | `python build_db.py soccer data/football-datasets` |
| MLB (FanGraphs) | Export the injury report as CSV, one per season | `python build_db.py mlb-fangraphs data/fg_2023.csv --season 2023` |
| NBA live (ESPN) | (scraped) | `python build_db.py espn-nba` — run regularly to build a history |
| NCAA ISP, Covers, anything else | Your CSV | `python build_db.py import-csv FILE --source-id 3 --table injuries --sport Multi-sport --level College --map "Their Column=injury_date,Other=body_part"` |

Kaggle downloads need an API token (`~/.kaggle/kaggle.json`, from your Kaggle account settings). Re-running any loader replaces that source's rows, so reloading is safe.

## Pull data in Python

```python
import sportsdb as sdb

sdb.tables()                                  # what's loaded
nba = sdb.injuries(sport="NBA", seasons=["2016-17", "2017-18"])
knees = sdb.injuries(body_part="Knee")
nfl_big = sdb.contracts(sport="NFL", min_apy=30_000_000)
model_df = sdb.contracts_with_injuries(sport="NFL")   # contracts + career injury totals
sdb.body_part_rates()                         # frequency and avg days missed by body part
sdb.insurance_cases()

sdb.query("SELECT team, COUNT(*) n FROM injuries WHERE sport='NBA' GROUP BY team ORDER BY n DESC")
sdb.query("SELECT * FROM injuries WHERE team = :team", {"team": "Lakers"})   # :named parameters
```

## Tables

- **injuries** — every sport in one shape. `record_type` tells you what a row is: `event` (one injury with start/return dates), `weekly_report` (NFL weekly status line), or `snapshot` (a live-page reading on a given date).
- **contracts** — whole contracts (NFL: value, APY, guaranteed) and per-season salaries (NBA). All amounts are full US dollars.
- **market_values** — soccer market values (EUR).
- **insurance_cases** — Watson plus blank rows for Rodgers, Tua, Goff, Burrow, McCaffrey to fill in.
- **sources** — the registry; `status` flips to `Collected` when a loader runs.
- **raw_*** — each source exactly as downloaded, in case the normalized version drops something you need.
- **v_injury_summary**, **v_contracts_with_injuries** — ready-made views for modeling.

## Things to know

- `injury_date`, `snapshot_date` and `value_date` are real MySQL `DATE` columns, so they come back as Python dates (use `pd.to_datetime(df.injury_date)` for date math). `return_date` stays text because ESPN gives values like "Oct 22".
- `player_key` is a normalized name (lowercase, no punctuation or Jr./III) used to join across sources. Name matching isn't perfect; spot-check joins for common names.
- `body_part` is guessed from the injury description by keyword. Check it before relying on it.
- NBA salary seasons are single years (`2012`) while NBA injury seasons are `2011-12`; the provided views join on player, not season.
- Column names in the Kaggle, soccer, and FanGraphs files were matched flexibly, but these loaders were tested on sample files, not the real downloads. If one fails, the error lists the columns it found so you can adjust the `pick(...)` names.
- The ESPN scraper depends on ESPN's page layout and may need updating if it changes.
