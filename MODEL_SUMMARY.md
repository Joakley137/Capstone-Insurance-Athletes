# Injury Model Results

Two GLMs price injury risk for an athlete:
1. **Severity**: days missed (gamma GLM)
2. **Career-ending**: probability the injury ends the career (logistic GLM)

## Key findings

- **The injury itself drives time out.** Grouping injuries into 16 severity tiers cuts the error in predicted days out by a quarter (33 vs. 44 days, mean absolute error). Achilles and knee tears keep a player out about 4× as long as a typical hamstring injury; illness about a quarter as long.
- **Career-ending injuries are rare but predictable.** About 1 in 240 injuries ends a career. The model ranks them well (AUC 0.86): the riskiest 10% of injuries contain 55% of the career-ending ones.
- **Knee and Achilles injuries, surgery, and age raise career-ending risk most.** Knee/Achilles injuries carry about 9–10× the odds of a hamstring injury, surgery or a tear about 3×, and each extra year of age about 1.27×.
- **Regular players are far less likely to retire after an injury.** More games in the past 12 months sharply lowers the odds.
- **Example:** a torn ACL means about 160–175 expected days out; the chance it ends the career is about 2% for a 27-year-old regular starter and about 9% for a typical 33-year-old.

Data: Transfermarkt professional soccer injuries, player profiles and playing time (163,414 injuries, 34,429 players). Models are trained on 80% of players and tested on the other 20%; all of a player's injuries stay on one side. Each GLM is checked against a no-predictor baseline.

---

## GLM specification

```text
# Severity: days missed
days_missed ~ severity_tier + position + age + age² + log(1+prior injuries)
              + reinjury_60d + prior_same_part
family = Gamma, link = log            # coefficients -> exp(coef) multipliers

# Career-ending: yes/no
career_ending ~ body_region + injury_type + position + age + age² + log(1+prior injuries)
                + reinjury_60d + prior_same_part + log(1+games, 12m) + no_games_12m
                + log(1+career games) + year
family = Binomial, link = logit       # coefficients -> exp(coef) odds ratios
```

- Baseline: unspecified hamstring injury, 26-year-old midfielder.
- No market value and no interactions in either model.
- Rare injury groups are pooled (severity: ≥300 injuries per group; career-ending: ≥20 events, else "Other").
- Severity: 57 injury groups merged into 16 tiers (adjacent groups merged unless significantly different at 5%, decided on training players only).
- Fit with `statsmodels` `sm.GLM` in `severity.py` and `career.py`.

---

## Accuracy (test players)

![Model accuracy vs. baseline](figures/model_accuracy.png)

| | Baseline | GLM |
|---|---|---|
| Severity: mean absolute error | 44.0 days | **33.3 days** |
| Severity: gamma deviance | 1.391 | **0.929** |
| Severity: mean predicted vs. actual (47.5) | 48.2 | 48.3 |
| Severity: share above predicted 90th pct (target 10%) | 10.6% | 8.9% |
| Career-ending: AUC | 0.50 | **0.857** |
| Career-ending: riskiest 10% capture | — | **55%** of career-ending injuries |
| Career-ending: mean predicted vs. actual | — | 0.40% vs. 0.46% |

Career-ending rate: 504 of 120,682 injuries (0.42%, about 1 in 240).

## Severity drivers (multipliers on expected days)

| Predictor | Multiplier |
|---|---|
| **Severity tier** | 0.27× (illness) to 4.19× (knee/Achilles tears) vs. hamstring |
| Same body part injured before | 1.09× |
| Re-injury within 60 days of return | 1.03× |
| Goalkeeper / attacker (vs. midfielder) | 1.04× / 0.98× |
| log(1 + prior injuries) | 0.84× per unit (likely a reporting effect) |
| Age | small, U-shaped |

**Typical length by injury (data):**

| Injury | Mean days | Median days |
|---|---|---|
| Achilles tear / rupture | 203 | 190 |
| Knee tear / rupture (mostly ACL) | 190 | 191 |
| Knee surgery | 167 | 150 |
| Lower-leg fracture | 152 | 130 |
| Foot fracture | 83 | 74 |
| Ankle injury (unspecified) | 44 | 25 |
| Hamstring injury (unspecified) | 43 | 29 |
| Hamstring strain | 32 | 23 |
| Concussion | 21 | 11 |
| Illness | 11 | 8 |

## Severity tiers (pricing table)

Each tier differs significantly from its neighbours. Full mapping: `severity_models.ipynb`, Severity tiers.

![Severity tier multipliers](figures/severity_tiers.png)

| Tier | Multiplier vs. Tier 8 | Median days | Examples |
|---|---|---|---|
| 1 | 0.27 | 8 | Illness |
| 2 | 0.48 | 11 | Concussion, knee/trunk knocks |
| 3 | 0.55 | 11 | Foot and other knocks |
| 4 | 0.63 | 17 | Ankle ligament, muscle strains, thigh |
| 5 | 0.75 | 22 | Calf and hamstring strains, head fracture |
| 6 | 0.84 | 21 | Groin, muscle tears |
| 7 | 0.91 | 21 | Arm/hand, back/neck, trunk |
| 8 | 1.00 | 28 | Hamstring/ankle (unspecified), calf tear |
| 9 | 1.07 | 29 | Arm/hand fracture, foot, groin tear |
| 10 | 1.29 | 33 | Achilles tendon, ankle tear, hip, shoulder |
| 11 | 1.47 | 34 | Knee (unspecified), shoulder fracture, thigh tear |
| 12 | 1.82 | 62 | Foot fracture, groin surgery |
| 13 | 2.38 | 63 | Knee ligament |
| 14 | 2.79 | 99 | Ankle surgery, other fractures and surgeries |
| 15 | 3.54 | 138 | Knee surgery, lower-leg fracture |
| 16 | 4.19 | 192 | Achilles and knee tears |

## Career-ending drivers (odds ratios)

![Career-ending risk factors](figures/career_risk_factors.png)

| Predictor | Odds ratio |
|---|---|
| **Knee / Achilles** (vs. hamstring) | **9.8× / 9.1×** (95% CI ~3–27) |
| **Surgery / tear** | **3.2× / 3.0×** |
| **Age** | 1.27× per year around 26 |
| Defender (vs. midfielder) | 1.44× |
| log(1 + games, last 12 months) | 0.53× (regular players much less likely) |
| log(1 + career games) | 0.80× |
| log(1 + prior injuries) | 0.73× |
| Year | 1.07× per year (likely data coverage) |

## Recommendations

- **Expected days out:** gamma GLM with severity tiers (multipliers = pricing table).
- **Career-ending risk:** main-effects logistic GLM.
- **Expected cost** ≈ P(career ends) × remaining contract + (1 − P) × expected days × daily salary. E.g. 1% on a $50M contract adds ~$500,000.

## Predicting a new injury

`predict.py` returns expected days, a range (median and 90th percentile), and P(career-ending). Torn cruciate ligament:

| Player | Expected days | Median / 90th pct | P(career ends) |
|---|---|---|---|
| 27-year-old defender, 2,500 min last year, 4 prior injuries | 160 | 157 / 275 | ~2.0% |
| 33-year-old defender, typical minutes and history | 173 | 170 / 298 | ~9.4% |

## Limitations

- **Soccer only**: fitted on professional soccer; results may not carry over to other sports.
- **Patterns, not causes**: e.g. more recorded injuries go with *shorter* ones, likely reporting differences across leagues.
- **Career-ending is inferred** from appearances; a move to amateur football counts.
- **Body part is keyword-matched**; 18% are "unknown injury".

---

*Notebooks: `severity_models.ipynb` (days missed), `career_ending_model.ipynb` (career-ending); coefficients on all data. Code: `severity.py`, `career.py` (held-out metrics: `python severity.py`, `python career.py`), `predict.py` (new injuries).*
