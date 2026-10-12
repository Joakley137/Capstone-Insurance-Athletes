# Sports Injury & Contract Database

A MySQL database (`sports_injury`) of athlete injuries, contracts and market values, plus models that predict how long an injury keeps a player out and whether it ends his career. Results in plain language: **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)**.

## Files

| File | What it is |
|---|---|
| [README.md](README.md) | This page: setup, loading the data, the database schema, the models and how to run them. |
| [MODEL_SUMMARY.md](MODEL_SUMMARY.md) | The model results and most important predictors, with a plain-language explanation for readers new to modeling. |
| [build_db.py](build_db.py) | Creates the database tables and loads each data source you download (NBA, NFL, soccer, MLB, ESPN, any CSV). It doesn't download anything on its own. |
| [sportsdb.py](sportsdb.py) | Pulls data back out of the database as pandas tables: `sdb.injuries(...)`, `sdb.contracts(...)`, `sdb.players(...)` and so on. |
| [severity.py](severity.py) | The severity model: how many days an injury keeps a player out. Builds the modeling table, the rating classes and severity tiers, and fits and compares the gamma GLM and alternatives. |
| [frequency.py](frequency.py) | The injury-frequency model: how often a player gets injured, per day at risk, from age, position, playing time and injury history (Poisson and negative binomial GLMs). Multiply by the severity model for expected days lost. |
| [premium_test.py](premium_test.py) | Tests what a club's insurance premium could be paying for: wages while players are out (with waiting periods), or the rest of a contract after a career-ending injury. Compares the premium to realized and model-expected losses on each club's squad. |
| [policy_sim.py](policy_sim.py) | Simulates a club's seasons to compare deductible policies: the club's average and worst-year cost against the insurer's profit. |
| [club_financials.csv](club_financials.csv) | Gross payroll and total insurance premium per club and season (Lyon, Real Madrid, AC Milan, Dortmund). Input to `premium_test.py` and `policy_sim.py`. |
| [career.py](career.py) | The career-ending model: the chance an injury ends the player's career (logistic regression), with its comparison models. |
| [predict.py](predict.py) | Predictions for injuries you describe: expected days out, a likely range and the career-ending chance, for one injury or a whole CSV. |
| [example_injuries.csv](example_injuries.csv) | Example input for `predict.py --csv`, showing the columns it accepts. |
| [explore.ipynb](explore.ipynb) | Short notebook that connects to the database and shows a sample of each table. |
| [frequency_model.ipynb](frequency_model.ipynb) | The frequency model step by step: the player-season table, the simple "earlier injury" comparison, model comparison, calibration and relativities. |
| [severity_models.ipynb](severity_models.ipynb) | The severity model step by step, with charts: data, rating classes, interaction search, model comparison, relativities, severity tiers and conclusions. |
| [career_ending_model.ipynb](career_ending_model.ipynb) | The career-ending model step by step: how "career-ending" is defined and checked, model comparison, calibration, odds ratios and conclusions. |
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

Three models price an injury: how often it happens (**frequency**), how long it keeps a player out (**severity**), and whether it ends his career (**career-ending**). **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)** gives the results, the most important predictors and a plain-language explanation for readers new to modeling. Soccer is the only source loaded so far that records how long injuries lasted, so both are fitted on soccer. Each notebook runs the model with plots and explains the results.

### Severity: days missed — `severity.py`, `severity_models.ipynb`

```bash
python severity.py                 # model comparison and gamma GLM multipliers (~2 min)
python severity.py --select        # also re-run the interaction search (~6 min more)
```
```python
import severity as sev
data = sev.build_dataset()                 # one row per injury, with predictors and a rating class
train, test = sev.split(data)              # by player, so no player is in both
models = sev.fit_all(train)
sev.compare(models, test)                  # log-likelihood, deviance, MAE/RMSE, 90th-percentile check
sev.relativities(models["Gamma GLM, rating classes + interactions"])   # exp(coef) = multiplier on days
sev.class_table(data)                      # the rating classes with their sizes and average days
```

- **Rating classes.** Each injury gets a class: body region × injury type (tear / fracture / surgery / ligament / strain / knock…). Combinations with fewer than 300 injuries are pooled, first by injury type ("Fracture - other sites"), then by body region, then into "Other (rare)". This gives 57 classes, each large enough for a credible average.
- **Other predictors:** age and age²; position; prior injuries; re-injury within 60 days; same body part injured before; market value at the time of injury; **minutes played in the 12 months before the injury** and career minutes; year; offseason. Minutes count only seasons that had ended before the injury, so games played after it can't leak in.
- **Interactions** are chosen by forward selection on a validation split of the training data: year × body region, market value × injury type, and minutes × body region.
- **Severity tiers.** For the pricing table, the 57 rating classes are merged into **11 severity tiers**: neighbouring classes are merged until every pair of adjacent tiers differs significantly (p < 0.05). The tiers are built on training data only. Test accuracy is unchanged (deviance 0.9007 vs. 0.9001). Use `models["Gamma GLM, severity tiers + interactions"].tier_table(train)` to see them.
- **Models compared:**
  - gamma GLM, with main effects only, with rating classes + interactions, and with severity tiers + interactions
  - inverse Gaussian GLM
  - lognormal
  - Weibull AFT, which also uses injuries still open on the data date
  - gradient boosting, as an accuracy benchmark
  - a no-predictor baseline
- **Findings:**
  - **Rating classes:** they were the biggest improvement. Together with the interactions, they cut test deviance from 0.922 to 0.900 and closed half the gap to gradient boosting.
  - **Expected cost:** the gamma GLM with severity tiers is the model for this; its relativities are the rating table.
  - **Tail:** the lognormal still fits the shape of the distribution best, so use it for tail-based pricing.

### Career-ending: yes/no — `career.py`, `career_ending_model.ipynb`

```bash
python career.py
```
```python
import career
data = career.build_dataset()              # one row per injury, career_ending = 0/1
train, test = career.split(data)
models = career.fit_all(train)
career.compare(models, test)               # AUC, average precision, log loss, Brier, top-10% capture
career.odds_ratios(models["Logistic GLM, main effects"])
```

- **Definition:** an injury is career-ending when the player never made another recorded professional appearance after it, and hadn't returned by the end of that season. An injury is only labeled once two complete seasons of data follow it. That leaves 504 career-ending injuries out of 120,682 (0.42%). Of those players, 84% are listed as retired or without a club today.
- **Predictors** are only what is known on the day of the injury; how long the injury lasted is not used.
- **Result:** the main-effects logistic GLM has an AUC of 0.87, and its riskiest 10% of injuries contain 61% of the career-ending ones. Interactions and gradient boosting did not beat it on held-out data.
- **Biggest risk factors:** knee or Achilles injuries (about 8× the odds of a hamstring injury), surgery or a tear (about 3×), age, little recent playing time, and playing outside the leagues that have market values.

### Frequency: how often — `frequency.py`, `frequency_model.ipynb`

```bash
python frequency.py                 # top-5 leagues from 2012: recurrence table and model comparison
python frequency.py --all-leagues   # every league in the data
```

- **Table:** one row per player and July-June season, with an injury count and the days at risk. Seasons with no injuries are included. Only players with an appearance in a top-5 league are kept (`competition_id` GB1, ES1, IT1, L1, FR1), and only seasons from 2012: injury records are thin before then (about 2% of players have one in a season up to 2006, 10-15% by 2012).
- **Models:** Poisson and negative binomial GLMs with days at risk as the offset, plus a gradient-boosting benchmark. Predictors are known on 1 July: earlier injuries, last season's injuries, still injured on 1 July, age, position, market value and minutes.
- **Result:** the Poisson GLM is well calibrated (1.006 predicted vs 1.018 actual injuries a season) and gradient boosting adds almost nothing. The injury rate rises with earlier injuries (about x1.3 for one, x2.6 for nine), but how long the earlier injuries lasted adds nothing. Goalkeepers have about 28% fewer injuries. The all-leagues table is not used: players Transfermarkt barely tracks make "no injury history" look safer than it is.
- **With severity:** expected injuries x expected days per injury is the expected days lost. `model.p_injury(df)` gives the chance of at least one injury in a season.

### What do club premiums pay for? — `premium_test.py`, `policy_sim.py`

```bash
python premium_test.py              # a few minutes: refits the models on all the data
python policy_sim.py --min-wage 8   # simulated deductible policies (optional --min-wage)
```

Compares four clubs' insurance premiums (`club_financials.csv`, 49 club-seasons) with the loss their squads would cause, as a share of payroll. A player's wage is his market-value share of the payroll, because the data has no wages. Findings:
- **Not all time out.** Paying full wages for every injury day would cost 6-13x the premium (48x for Dortmund).
- **Not career-ending cover alone.** With three years of contract left it is 12-20% of the premium for Milan, Madrid and Lyon (30 career-ending events behind the rate).
- **Size fits only very long absences:** a waiting period of roughly 180-270 days if the whole squad is insured, which is about a cruciate rupture. The data cannot separate that from a lower deductible with a loading, or from only some players being insured.
- **Insuring only the best-paid players did not fit better** (wage threshold or top-N players), and the premium does not follow each club's realized injuries from year to year. Differences between clubs (Dortmund pays about a quarter of the others' rate) are not explained by injury risk, so something else sets them: league or country rules, negotiated terms, or what "total insurance premium" includes.

### Predicting your own injuries — `predict.py`

Describe an injury and get expected days out, a realistic range and the chance it ends the player's career:

```bash
python predict.py --injury "Cruciate ligament tear" --age 27 --position Defender \
                  --market-value 15000000 --minutes-last-12-months 2500 --career-minutes 15000 --prior-injuries 4
```
```
Read as: Tear / rupture — Knee  (rating class 'Knee - Tear / rupture', Tier 12 of 12, higher = longer)
Expected time out: 188 days
Likely range: half of similar injuries take under 188 days, 3 in 4 under 250, 9 in 10 under 319
Chance of missing more than 90 days: 76%;  more than 180 days: 53%
Chance this injury ends the career: 0.47%  (average injury: about 0.42%)
Filled in with typical values: nothing
```

- **Many injuries at once:** `python predict.py --csv example_injuries.csv --out predictions.csv`. `example_injuries.csv` shows the columns; only `age` and `injury` are required.
- **From Python:** `predict.injury("Broken foot", age=30, position="Midfield")` returns the same numbers as a Series.
- **Describing the injury:** use words, the way Transfermarkt does ("Hamstring strain", "Broken foot"), or pass `--body-part` / `--injury-type` directly. The output shows how the description was read, so you can check it.
- **Missing inputs** get a typical value (the median player), and the output lists what was filled in.
- **Market value:** a market value of 0 means "no market value", i.e. a lower-league player. That raises the career-ending risk about 4×.
- **Where the range comes from:** how actual injuries in the same severity tier spread around the model's prediction. On held-out players, about half fall below the predicted median and 10% above the 90th percentile, in every tier.
- **Fitting:** the first run fits the models on all the data and saves them to `models/` (about a minute; not committed). Later runs are instant. After reloading data, run `python predict.py --refit`. Fitted on all the data rather than the 80% training split, the tier merge finds 12 tiers rather than 11.

## Things to know

- `injury_date`, `snapshot_date` and `value_date` are real MySQL `DATE` columns, so they come back as Python dates (use `pd.to_datetime(df.injury_date)` for date math). `return_date` stays text because ESPN gives values like "Oct 22".
- `player_key` is a normalized name (lowercase, no punctuation or Jr./III) used to join across sources. Name matching isn't perfect; spot-check joins for common names.
- `body_part` is guessed from the injury description by keyword. Check it before relying on it.
- Transfermarkt's `minutes_played` column in `player_performances.csv` is actually **minutes per goal** (blank when the player didn't score). The loader turns it into real minutes (minutes per goal × goals) and, for seasons with no goals, estimates minutes from appearances: 90 per full game, 73.5 when subbed off, 18.5 when subbed on. On seasons where minutes are known, that estimate is within 3% (median). `minutes_estimated` marks which is which. The 25/26 season was only partly downloaded.
- NBA salary seasons are single years (`2012`) while NBA injury seasons are `2011-12`; the provided views join on player, not season.
- Column names in the Kaggle, soccer, and FanGraphs files were matched flexibly, but these loaders were tested on sample files, not the real downloads. If one fails, the error lists the columns it found so you can adjust the `pick(...)` names.
- The ESPN scraper depends on ESPN's page layout and may need updating if it changes.
