# Model Equations

The three fitted GLMs behind a season quote, written out with their coefficients, and how they combine. These are the all-data fits that `python -m model.predict` uses (refit 2026-10-10; `--refit` changes them slightly). All three use a log or logit link, so **exp(coef)** is the multiplier on the rate, the expected days or the odds.

Notation: ln is the natural log, ln(1 + x) keeps zeros defined, and categorical predictors are 0/1 indicators (1 if the player is in that group). Each table lists every term in the model; a reference group has no row of its own and is absorbed into the intercept.

---

## 1. Frequency: expected injuries per season (negative binomial GLM)

$$
\ln \lambda = \beta_0 + \beta_{\text{league}} + \beta_{\text{position}} + \beta_1 a + \beta_2 a^2 + \beta_3 \ln(1+g) + \beta_4 Z + \beta_5 \ln(1+n) + \beta_6 \ln(1+d)
$$

$$
N \sim \text{NegBin}(\text{mean } \lambda,\ \text{variance } \lambda + \alpha\lambda^2), \qquad \alpha = 0.223
$$

Fitted on 30,240 big-five player-seasons (2015/16 to 2024/25). **Reference player:** Premier League midfielder aged 26.

| Symbol | Predictor | What it is | Coef | Rate ratio exp(coef) | p-value |
|---|---|---|---|---|---|
| β₀ | Intercept | Reference player with every numeric predictor at 0 | −1.4086 | 0.245 | <0.001 |
| β_league | LaLiga | 1 if the season is in LaLiga | −0.0279 | 0.97 | 0.19 |
| | Serie A | 1 if Serie A | 0.2922 | 1.34 | <0.001 |
| | Bundesliga | 1 if Bundesliga | 0.3251 | 1.38 | <0.001 |
| | Ligue 1 | 1 if Ligue 1 | −0.1235 | 0.88 | <0.001 |
| β_position | Goalkeeper | 1 if goalkeeper | −0.4278 | 0.65 | <0.001 |
| | Defender | 1 if defender | 0.0468 | 1.05 | 0.004 |
| | Attacker | 1 if attacker | 0.0898 | 1.09 | <0.001 |
| β₁ | a = age − 26 | Age at season start, centred at 26 | −0.0073 | 0.993 | <0.001 |
| β₂ | a² | Age curvature (rate peaks around 25) | −0.0037 | 0.996 | <0.001 |
| β₃ | ln(1 + g) | g = full games last season (minutes / 90, all competitions) | 0.2098 | 1.23 | <0.001 |
| β₄ | Z | 1 if no minutes last season | −0.0202 | 0.98 | 0.78 |
| β₅ | ln(1 + n) | n = injuries recorded before this season | 0.4127 | 1.51 | <0.001 |
| β₆ | ln(1 + d) | d = days out to injuries that started last season | 0.0778 | 1.08 | <0.001 |

**Example:** a 27-year-old Premier League defender, 2,500 minutes last season (g = 27.8), 4 prior injuries, 0 days out:
ln λ = −1.4086 + 0.0468 − 0.0073(1) − 0.0037(1) + 0.2098 ln(28.8) + 0.4127 ln(5) = −0.004, so **λ = 0.996 injuries**.

---

## 2. Severity: expected days out per injury (gamma GLM)

$$
\ln \mu = \beta_0 + \beta_{\text{tier}} + \beta_{\text{league}} + \beta_{\text{position}} + \beta_1 a + \beta_2 a^2 + \beta_3 \ln(1+n) + \beta_4 R + \beta_5 S
$$

$$
\text{days} \sim \text{Gamma}(\text{mean } \mu), \text{ log link}
$$

Fitted on 31,010 injuries in insured (big-five) seasons. **Reference injury:** Tier 7 (contains hamstring injuries), Premier League midfielder aged 26.

| Symbol | Predictor | What it is | Coef | Multiplier on days exp(coef) | p-value |
|---|---|---|---|---|---|
| β₀ | Intercept | Reference injury: about 52 days for a player with no prior injuries | 3.9470 | 51.8 days | <0.001 |
| β_tier | Tier 1 | Illness | −1.1986 | 0.30 | <0.001 |
| | Tier 2 | Knocks, head (unspecified) | −0.9315 | 0.39 | <0.001 |
| | Tier 3 | Foot/knee knocks, concussion | −0.6521 | 0.52 | <0.001 |
| | Tier 4 | Ankle ligament, back/neck, calf and muscle strains, thigh | −0.4541 | 0.64 | <0.001 |
| | Tier 5 | Calf, groin (unspecified), head fracture | −0.3019 | 0.74 | <0.001 |
| | Tier 6 | Ankle (unspecified), hamstring strain, hip, arm/hand fracture | −0.1268 | 0.88 | <0.001 |
| | *Tier 7* | *Reference: hamstring (unspecified), Achilles tendon, calf/groin/muscle tears, foot* | — | 1.00 | — |
| | Tier 8 | Knee (unspecified), shoulder | 0.1379 | 1.15 | <0.001 |
| | Tier 9 | Ankle tear, thigh tear, shoulder fracture, knee tendon | 0.3888 | 1.48 | <0.001 |
| | Tier 10 | Foot fracture, groin surgery | 0.6441 | 1.90 | <0.001 |
| | Tier 11 | Knee ligament, other surgeries | 0.8484 | 2.34 | <0.001 |
| | Tier 12 | Ankle surgery, lower-leg and other fractures | 1.1515 | 3.16 | <0.001 |
| | Tier 13 | Knee tear (mostly ACL), Achilles tear, knee surgery | 1.5454 | 4.69 | <0.001 |
| β_league | LaLiga / Serie A / Bundesliga / Ligue 1 | 1 if that league (vs Premier League); mostly recording practice | −0.1756 / −0.3752 / −0.3172 / −0.1362 | 0.84 / 0.69 / 0.73 / 0.87 | all <0.001 |
| β_position | Attacker / Defender / Goalkeeper | 1 if that position (vs midfielder) | −0.0143 / 0.0371 / 0.0937 | 0.99 / 1.04 / 1.10 | 0.44 / 0.03 / 0.006 |
| β₁ | a = age − 26 | Age at injury, centred at 26 | 0.0028 | 1.003 | 0.21 |
| β₂ | a² | Age curvature | 0.0009 | 1.001 | 0.006 |
| β₃ | ln(1 + n) | n = injuries before this one | −0.1144 | 0.89 | <0.001 |
| β₄ | R | 1 if within 60 days of returning from the last injury | 0.0445 | 1.05 | 0.005 |
| β₅ | S | 1 if the same body part was injured before | 0.0622 | 1.06 | <0.001 |

Every injury type maps to one tier; the full mapping is `tier_of` in the fitted model (and the table in `runs/severity_models.ipynb` for the all-league version). The chance an injury lasts beyond the deferment, q = P(days > D), uses how actual days spread around μ within each tier (not the gamma shape alone, which fits the far tail poorly).

---

## 3. Career-ending: probability an injury ends the career (logistic GLM)

$$
\ln\frac{p}{1-p} = \beta_0 + \beta_{\text{region}} + \beta_{\text{type}} + \beta_{\text{position}} + \beta_1 a + \beta_2 a^2 + \beta_3 \ln(1+n) + \beta_4 R + \beta_5 S + \beta_6 \ln(1+g_{12}) + \beta_7 Z_{12} + \beta_8 \ln(1+G) + \beta_9 t
$$

Fitted on 120,682 injuries in all leagues, 504 career-ending. **Reference injury:** hamstring, unspecified type, midfielder aged 26, year 2015.

| Symbol | Predictor | What it is | Coef | Odds ratio exp(coef) | p-value |
|---|---|---|---|---|---|
| β₀ | Intercept | Reference injury with every numeric predictor at 0 | −4.6377 | 0.0097 | <0.001 |
| β_region | Achilles | 1 if Achilles (vs hamstring) | 2.3904 | 10.9 | <0.001 |
| | Knee | 1 if knee | 2.3896 | 10.9 | <0.001 |
| | Unknown | 1 if body part not recorded | 1.4244 | 4.16 | 0.005 |
| | Other | 1 if any other body part | 0.6327 | 1.88 | 0.22 |
| β_type | Surgery | 1 if surgery (vs unspecified type) | 1.1935 | 3.30 | <0.001 |
| | Tear / rupture | 1 if tear or rupture | 1.0771 | 2.94 | <0.001 |
| | Fracture | 1 if fracture | 0.8878 | 2.43 | <0.001 |
| | Ligament / joint | 1 if ligament or joint | 0.4423 | 1.56 | 0.02 |
| | Other | 1 if any other type | −0.7351 | 0.48 | <0.001 |
| β_position | Attacker / Defender / Goalkeeper | 1 if that position (vs midfielder) | −0.2853 / 0.2919 / 0.1437 | 0.75 / 1.34 / 1.16 | 0.03 / 0.01 / 0.38 |
| β₁ | a = age − 26 | Age at injury, centred at 26 | 0.2539 | 1.29 per year | <0.001 |
| β₂ | a² | Age curvature | −0.0040 | 0.996 | 0.02 |
| β₃ | ln(1 + n) | n = injuries before this one | −0.2955 | 0.74 | <0.001 |
| β₄ | R | 1 if within 60 days of returning from the last injury | −0.0838 | 0.92 | 0.52 |
| β₅ | S | 1 if the same body part was injured before | 0.1454 | 1.16 | 0.26 |
| β₆ | ln(1 + g₁₂) | g₁₂ = full games in the 12 months before the injury | −0.6049 | 0.55 | <0.001 |
| β₇ | Z₁₂ | 1 if no games in the last 12 months | −0.4040 | 0.67 | 0.05 |
| β₈ | ln(1 + G) | G = career full games before the injury | −0.2481 | 0.78 | <0.001 |
| β₉ | t = year − 2015 | Year of injury (data coverage; capped at 2023 when predicting) | 0.0666 | 1.07 per year | <0.001 |

---

## 4. Combining them into season probabilities

For a player, a random future injury has an unknown type, so q and c average the severity and career models over the injury mix (w_k = share of insured injuries in body-region × type cell k):

$$
q = \sum_k w_k \, P(\text{days} > D \mid k,\ \text{player}), \qquad c = \sum_k w_k \, p(k,\ \text{player})
$$

Thinning the negative binomial count (each injury independently covered with probability q, or career-ending with probability c) gives, for the season:

$$
P(\text{any injury}) = 1 - (1 + \alpha\lambda)^{-1/\alpha}
$$

$$
P(\text{covered injury}) = 1 - (1 + \alpha\lambda q)^{-1/\alpha}
$$

$$
P(\text{career-ending injury}) = 1 - (1 + \alpha\lambda c)^{-1/\alpha}
$$

with α = 0.223 and D = 60 days by default. For the 27-year-old defender above: λ = 0.996, q = 0.174, c = 0.0025, so P(any) = 59%, P(covered) = 16% and P(career-ending) = 0.25%.

---

*Code: `model/frequency.py`, `model/severity.py`, `model/career.py`, `model/pricing.py`. Print these tables with `frequency.coefficients`, `severity.coefficients` and `career.coefficients` on `predict.models()`.*
