# Sports Injury & Contract Database

A MySQL database (`sports_injury`) of athlete injuries, contracts and market values, plus models that predict how long an injury keeps a player out and whether it ends his career. Results in plain language: **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)**.

## Files

| File | What it is |
|---|---|
| [README.md](README.md) | This page: setup, loading the data, the database schema, the models and how to run them. |
| [MODEL_SUMMARY.md](MODEL_SUMMARY.md) | The model results and most important predictors, with a plain-language explanation for readers new to modeling. |
| [build_db.py](build_db.py) | Creates the database tables and loads each data source you download (NBA, NFL, soccer, MLB, ESPN, any CSV). It doesn't download anything on its own. |
| [sportsdb.py](sportsdb.py) | Pulls data back out of the database as pandas tables: `sdb.injuries(...)`, `sdb.contracts(...)`, `sdb.players(...)` and so on. |
| [severity.py](severity.py) | The severity model: how many days an injury keeps a player out. Builds the modeling table, the rating classes and severity tiers, and fits the tiered gamma GLM against a no-predictor baseline. |
| [career.py](career.py) | The career-ending model: the chance an injury ends the player's career (logistic regression), compared with the base rate. |
| [predict.py](predict.py) | Predictions for injuries you describe: expected days out, a likely range and the career-ending chance, for one injury or a whole CSV. |
| [example_injuries.csv](example_injuries.csv) | Example input for `predict.py --csv`, showing the columns it accepts. |
| [explore.ipynb](explore.ipynb) | Short notebook that connects to the database and shows a sample of each table. |
| [severity_models.ipynb](severity_models.ipynb) | The severity model step by step, with charts: data, rating classes, severity tiers, model comparison, relativities and conclusions. |
| [career_ending_model.ipynb](career_ending_model.ipynb) | The career-ending model step by step: how "career-ending" is defined and checked, comparison with the base rate, calibration, odds ratios and conclusions. |
| [requirements.txt](requirements.txt) | Python packages to install (`pip install -r requirements.txt`). |
| [.gitignore](.gitignore) | Keeps downloaded data (`data/`), the virtual environment and saved model files (`models/`) out of the repository. |

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
| Soccer (salimt) | `git clone https://github.com/salimt/football-datasets data/football-datasets`, then `cd data/football-datasets && git lfs pull` (needs `brew install git-lfs`; the appearances file is stored with Git LFS) | `python build_db.py soccer data/football-datasets` (injuries, market values, player profiles, appearances) |
| Soccer player profiles only | (same repo) | `python build_db.py soccer-players data/football-datasets` — birth dates, height, position, current club |
| Soccer appearances only | (same repo) | `python build_db.py soccer-seasons data/football-datasets` — appearances and minutes per season |
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
ages = sdb.injuries_with_age(sport="Soccer")  # injuries + age_at_injury, position_group, height_cm
sdb.players(player="messi")                   # player profiles
sdb.player_seasons(player_id="28003")         # appearances and minutes by season (Transfermarkt id)
sdb.insurance_cases()

sdb.query("SELECT team, COUNT(*) n FROM injuries WHERE sport='NBA' GROUP BY team ORDER BY n DESC")
sdb.query("SELECT * FROM injuries WHERE team = :team", {"team": "Lakers"})   # :named parameters
```

## Schema

Solid lines are foreign keys. Dotted lines are joins the views and `sportsdb.py` use but the database doesn't enforce: on `player_key` + `sport`, or for `players` and `player_seasons` on `ext_player_id` + `sport`.

```mermaid
erDiagram
    sources ||--o{ injuries : "source_id"
    sources ||--o{ contracts : "source_id"
    sources ||--o{ market_values : "source_id"
    sources ||--o{ players : "source_id"
    sources ||..o{ load_log : "source_id"
    sources ||--o{ player_seasons : "source_id"
    players ||..o{ injuries : "ext_player_id + sport"
    players ||..o{ player_seasons : "ext_player_id + sport"
    contracts }o..o{ injuries : "player_key + sport"
    contracts }o..o{ market_values : "player_key + sport"
    insurance_cases }o..o{ contracts : "player_key + sport"

    sources {
        INT source_id PK
        VARCHAR name
        TEXT url
        VARCHAR sport
        VARCHAR level
        VARCHAR coverage
        VARCHAR format
        TINYINT has_injury
        TINYINT has_contract
        TINYINT has_insurance
        TINYINT has_nil
        TEXT role_in_model
        TEXT access_notes
        VARCHAR status "Not started / Collected"
        TEXT notes
    }

    injuries {
        INT injury_id PK
        INT source_id FK
        VARCHAR record_type "event / weekly_report / snapshot"
        VARCHAR sport
        VARCHAR league
        VARCHAR level
        VARCHAR season
        INT week
        VARCHAR player_name
        VARCHAR player_key "normalized name"
        VARCHAR ext_player_id
        VARCHAR team
        VARCHAR position
        DATE injury_date
        VARCHAR return_date
        INT days_missed
        INT games_missed
        VARCHAR body_part
        TEXT injury_desc
        VARCHAR status
        DATE snapshot_date
        DATETIME loaded_at
    }

    contracts {
        INT contract_id PK
        INT source_id FK
        VARCHAR sport
        VARCHAR league
        VARCHAR level
        VARCHAR player_name
        VARCHAR player_key
        VARCHAR ext_player_id
        VARCHAR team
        VARCHAR position
        VARCHAR season "per-season salary rows"
        INT year_signed "whole-contract rows"
        INT years
        DOUBLE total_value "USD"
        DOUBLE apy "USD"
        DOUBLE guaranteed "USD"
        DOUBLE salary "USD"
        DATETIME loaded_at
    }

    market_values {
        INT mv_id PK
        INT source_id FK
        VARCHAR sport
        VARCHAR league
        VARCHAR player_name
        VARCHAR player_key
        VARCHAR ext_player_id
        DATE value_date
        DOUBLE market_value
        VARCHAR currency
        DATETIME loaded_at
    }

    players {
        INT player_id PK
        INT source_id FK
        VARCHAR sport
        VARCHAR player_name
        VARCHAR player_key
        VARCHAR ext_player_id "source's player id"
        DATE date_of_birth
        DOUBLE height_cm
        VARCHAR foot
        VARCHAR position
        VARCHAR position_group "Goalkeeper / Defender / Midfield / Attack"
        VARCHAR citizenship
        VARCHAR current_club "Retired / Without Club / club"
        DATE date_of_death
        DATETIME loaded_at
    }

    player_seasons {
        INT ps_id PK
        INT source_id FK
        VARCHAR sport
        VARCHAR ext_player_id
        VARCHAR season "23/24 or 2023"
        DATE season_start
        DATE season_end
        VARCHAR competition
        VARCHAR team
        INT squad_selections
        INT appearances
        INT subbed_in
        INT subbed_out
        INT goals
        DOUBLE minutes
        TINYINT minutes_estimated
        DATETIME loaded_at
    }

    insurance_cases {
        INT case_id PK
        VARCHAR player_name
        VARCHAR player_key
        VARCHAR sport
        VARCHAR team
        DOUBLE contract_total
        DOUBLE guaranteed
        TEXT injury
        VARCHAR seasons_affected
        INT games_missed
        DOUBLE reported_payout
        VARCHAR source_ids
        TEXT notes
    }

    load_log {
        INT source_id
        VARCHAR target
        INT rows
        TEXT file
        DATETIME loaded_at
    }
```

The views are built on top of these tables. `v_injury_summary` totals `injuries` per player and season. `v_contracts_with_injuries` adds each player's career totals from that summary to every `contracts` row. `v_injuries_with_age` adds the player's `age_at_injury`, position group and height to every injury. Every loader also saves its source file untouched in a `raw_<name>` table (e.g. `raw_nfl_injuries`). Those tables aren't in the diagram because their columns are whatever the source file has.

## Tables

- **injuries** — every sport in one shape. `record_type` tells you what a row is: `event` (one injury with start/return dates), `weekly_report` (NFL weekly status line), or `snapshot` (a live-page reading on a given date).
- **contracts** — whole contracts (NFL: value, APY, guaranteed) and per-season salaries (NBA). All amounts are full US dollars.
- **market_values** — soccer market values (EUR).
- **players** — player profiles: birth date, height, foot, position, current club, date of death (soccer, from Transfermarkt). Joins to `injuries` on `ext_player_id` + `sport`.
- **player_seasons** — appearances, substitutions, goals and minutes per player, season and competition (soccer). Used for playing time before an injury, and to tell whether a player ever played again.
- **insurance_cases** — Watson plus blank rows for Rodgers, Tua, Goff, Burrow, McCaffrey to fill in.
- **sources** — the registry; `status` flips to `Collected` when a loader runs.
- **raw_*** — each source exactly as downloaded, in case the normalized version drops something you need.
- **v_injury_summary**, **v_contracts_with_injuries**, **v_injuries_with_age** — ready-made views for modeling.

## Injury models

Two models price an injury: how long it keeps a player out (**severity**), and whether it ends his career (**career-ending**). **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)** gives the results, the most important predictors and a plain-language explanation for readers new to modeling. Soccer is the only source loaded so far that records how long injuries lasted, so both are fitted on soccer. Each notebook runs the model with plots and explains the results.

### Severity: days missed — `severity.py`, `severity_models.ipynb`

```bash
python severity.py                 # model comparison and tiered gamma GLM multipliers
```
```python
import severity as sev
data = sev.build_dataset()                 # one row per injury, with predictors and a rating class
train, test = sev.split(data)              # by player, so no player is in both
models = sev.fit_all(train)
sev.compare(models, test)                  # log-likelihood, deviance, MAE/RMSE, 90th-percentile check
glm = models["Gamma GLM, severity tiers"]
sev.relativities(glm)                      # exp(coef) = multiplier on days
glm.tier_table(train)                      # the severity tiers and the rating classes in each
sev.class_table(data)                      # the rating classes with their sizes and average days
```

- **Rating classes.** Each injury gets a class: body region × injury type (tear / fracture / surgery / ligament / strain / knock…). Combinations with fewer than 300 injuries are pooled, first by injury type ("Fracture - other sites"), then by body region, then into "Other (rare)", so each class is large enough for a credible average.
- **Severity tiers.** The rating classes are merged into severity tiers: neighbouring classes are merged until every pair of adjacent tiers differs significantly (p < 0.05). The tiers are built on training data only.
- **Model:** a gamma GLM (log link, main effects only) on severity tier, position, age and age², log(1 + prior injuries), re-injury within 60 days, and same body part injured before. Its relativities are the rating table.
- **Baseline:** a gamma model with no predictors, to show how much the predictors add.

### Career-ending: yes/no — `career.py`, `career_ending_model.ipynb`

```bash
python career.py
```
```python
import career
data = career.build_dataset()              # one row per injury, career_ending = 0/1
train, test = career.split(data)
models = career.fit_all(train)
career.compare(models, test)               # vs. the base rate: AUC, average precision, log loss, Brier, top-10% capture
career.odds_ratios(models["Logistic GLM, main effects"])
```

- **Definition:** an injury is career-ending when the player never made another recorded professional appearance after it, and hadn't returned by the end of that season. An injury is only labeled once two complete seasons of data follow it. That leaves 504 career-ending injuries out of 120,682 (0.42%). Of those players, 84% are listed as retired or without a club today.
- **Model:** a main-effects logistic GLM, compared with the base rate.
- **Predictors** are only what is known on the day of the injury: body region, injury type, position, age, prior injuries, re-injury, same body part before, minutes in the last 12 months, career minutes and year. How long the injury lasted is not used.
- **Result:** `career.compare` reports how well the GLM ranks injuries (AUC, and the share of career-ending injuries in its riskiest 10%) against the base rate.
- **Biggest risk factors:** knee or Achilles injuries, surgery or a tear, age, and little recent playing time. `career.odds_ratios` gives the sizes.

### Predicting your own injuries — `predict.py`

Describe an injury and get expected days out, a realistic range and the chance it ends the player's career:

```bash
python predict.py --injury "Cruciate ligament tear" --age 27 --position Defender \
                  --minutes-last-12-months 2500 --career-minutes 15000 --prior-injuries 4
```
```
Read as: Tear / rupture — Knee  (rating class 'Knee - Tear / rupture', Tier 16 of 17, higher = longer)
Expected time out: 160 days
Likely range: half of similar injuries take under 157 days, 3 in 4 under 214, 9 in 10 under 275
Chance of missing more than 90 days: 73%;  more than 180 days: 39%
Chance this injury ends the career: 2.03%  (average injury: about 0.42%)
Filled in with typical values: nothing
```

- **Many injuries at once:** `python predict.py --csv example_injuries.csv --out predictions.csv`. `example_injuries.csv` shows the columns; only `age` and `injury` are required.
- **From Python:** `predict.injury("Broken foot", age=30, position="Midfield")` returns the same numbers as a Series.
- **Describing the injury:** use words, the way Transfermarkt does ("Hamstring strain", "Broken foot"), or pass `--body-part` / `--injury-type` directly. The output shows how the description was read, so you can check it.
- **Missing inputs** get a typical value (the median player), and the output lists what was filled in.
- **Minutes in the last 12 months, career minutes and the date** only affect the career-ending chance; days out don't use them.
- **Where the range comes from:** how actual injuries in the same severity tier spread around the model's prediction. On held-out players, about half fall below the predicted median and 10% above the 90th percentile, in every tier.
- **Fitting:** the first run fits the models on all the data and saves them to `models/` (about a minute; not committed). Later runs are instant. After reloading data, run `python predict.py --refit`. Because it is fitted on all the data rather than the 80% training split, the tier merge can find a different number of tiers than `severity.py` reports.

## Things to know

- `injury_date`, `snapshot_date` and `value_date` are real MySQL `DATE` columns, so they come back as Python dates (use `pd.to_datetime(df.injury_date)` for date math). `return_date` stays text because ESPN gives values like "Oct 22".
- `player_key` is a normalized name (lowercase, no punctuation or Jr./III) used to join across sources. Name matching isn't perfect; spot-check joins for common names.
- `body_part` is guessed from the injury description by keyword. Check it before relying on it.
- Transfermarkt's `minutes_played` column in `player_performances.csv` is actually **minutes per goal** (blank when the player didn't score). The loader turns it into real minutes (minutes per goal × goals) and, for seasons with no goals, estimates minutes from appearances: 90 per full game, 73.5 when subbed off, 18.5 when subbed on. On seasons where minutes are known, that estimate is within 3% (median). `minutes_estimated` marks which is which. The 25/26 season was only partly downloaded.
- NBA salary seasons are single years (`2012`) while NBA injury seasons are `2011-12`; the provided views join on player, not season.
- Column names in the Kaggle, soccer, and FanGraphs files were matched flexibly, but these loaders were tested on sample files, not the real downloads. If one fails, the error lists the columns it found so you can adjust the `pick(...)` names.
- The ESPN scraper depends on ESPN's page layout and may need updating if it changes.
