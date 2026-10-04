# Injury Model Summary

This project uses past injuries to predict two things about a new one:

1. **How long will it keep the player out?** (the *severity* model, in days)
2. **Will it end the player's career?** (the *career-ending* model, a yes/no probability)

Together they estimate what an injury is likely to cost, which is what an insurer needs to price a policy on an athlete. Part 1 gives the results. Part 2 explains how the models work, for readers who don't build models themselves.

All numbers below come from professional soccer: 163,414 injuries to 34,429 players, from Transfermarkt. Soccer is the only source in the database that records how long each injury lasted.

---

## Part 1 — Results

### How accurate are the models?

The models were built on 80% of the players and tested on the other 20%, which they had never seen.

**How long an injury lasts**
- For the test players, the predicted length of an injury was off by **33 days** on average.
- Giving every injury the same overall-average guess is off by **44 days**, so the model cuts the error by about a quarter.
- It is right on average: the test injuries lasted 47.5 days on average, and the model predicted 48.2.

**Whether an injury ends a career**
- Career-ending injuries are rare: about **1 in 240** injuries (0.42%).
- The model is good at spotting them. The 10% of injuries it rates as riskiest contain **61%** of all the career-ending ones.
- Shown one career-ending injury and one ordinary one, the model gives the career-ending one the higher risk **87%** of the time.

### What matters most: how long the player is out

Each group of predictors was removed in turn and the model refit without it. The bigger the loss in accuracy, the more that group matters.

| Rank | Predictor | Accuracy lost without it | What it shows |
|---|---|---|---|
| 1 | **Injury type and location** | 11.0% | Far more important than everything else. A knee tear is expected to last about **3.8×** as long as a typical hamstring injury; an illness about **0.27×** as long. |
| 2 | **Injury history** | 1.3% | Hurting the same body part again adds about **8%**. Getting hurt within 60 days of coming back adds about **5%**. |
| 3 | **Market value** | 0.8% | Tears in expensive players last about **6% longer** for each step up in value (each step is about 2.7× the value), likely because clubs bring stars back cautiously. Bruises and knocks run the other way. |
| 4 | **Year** | 0.6% | Injuries last about **2% longer** each year overall. Illnesses and hip injuries are the exception: they are getting shorter. |
| 5 | **Minutes played in the year before** | 0.6% | Players who had been playing a lot usually have slightly shorter injuries. **Achilles** and **lower-leg** injuries are the exception: more recent playing time goes with *longer* layoffs, consistent with overuse injuries. |
| 6 | **Injured in June or July** | 0.3% | These run about **28% longer**, because the time out includes the summer break. |
| 7 | Age | 0.1% | A small effect that peaks around age 31. |
| 8 | Position | ~0% | Almost no effect. |

**Typical length by injury, from the data:**

| Injury | Average days out | Median days out |
|---|---|---|
| Achilles tear / rupture | 203 | 190 |
| Knee tear / rupture (mostly cruciate ligament) | 190 | 191 |
| Knee surgery | 167 | 150 |
| Lower-leg fracture | 152 | 130 |
| Foot fracture | 83 | 74 |
| Ankle injury (unspecified) | 44 | 25 |
| Hamstring injury (unspecified) | 43 | 29 |
| Hamstring strain | 32 | 23 |
| Concussion | 21 | 11 |
| Illness | 11 | 8 |

The average is higher than the median because a few injuries run very long. That long tail is exactly what makes injuries expensive to insure.

### Severity tiers: the pricing table

The model starts with 57 injury groups, but many of them turn out to be statistically indistinguishable. Merging those leaves **11 severity tiers**, and the model is just as accurate with 11 as with 57. Each tier is clearly different from the ones next to it.

| Tier | Multiplier vs. Tier 6 | Median days out | Examples |
|---|---|---|---|
| 1 | 0.26 | 8 | Illness |
| 2 | 0.53 | 11 | Bruises and knocks to the knee or foot |
| 3 | 0.63 | 17 | Muscle strains, concussion, ankle ligament sprains, thigh knocks |
| 4 | 0.75 | 20 | Hamstring strains, calf and groin injuries |
| 5 | 0.87 | 21 | Unspecified injuries, muscle tears, calf and groin tears, head fractures |
| 6 | 1.00 | 28 | Hamstring and ankle injuries (unspecified), hand and arm fractures |
| 7 | 1.30 | 31 | Ankle tears, back, shoulder and unspecified knee injuries |
| 8 | 1.57 | 43 | Foot and shoulder fractures, groin surgery, hip injuries |
| 9 | 2.00 | 62 | Knee ligament injuries, other surgeries and tears |
| 10 | 2.96 | 141 | Achilles tears, knee and ankle surgery, lower-leg fractures |
| 11 | 3.61 | 192 | Knee tears (mostly cruciate ligament) |

The full list of which injury goes in which tier is in `severity_models.ipynb`, section 7.

### What matters most: whether the career ends

| Rank | Predictor | Accuracy lost without it | What it shows |
|---|---|---|---|
| 1 | **Injury location and type** | 7.5% | Knee and Achilles injuries carry about **8×** the risk of a hamstring injury. Surgery or a tear on top of that roughly **triples** the risk. |
| 2 | **Age** | 4.3% | Risk rises about **24% per year** around age 26, and keeps climbing more slowly after that. |
| 3 | **Market value** | 2.2% | Players with no market value (usually lower leagues) have about **4×** the risk. Dropping out of professional football is easier at that level. |
| 4 | **Minutes played in the year before** | 1.9% | Players who had been playing regularly are much less likely to see their career end. |
| 5 | Injury history, position, year | <0.1% each | Little effect once everything above is known. One exception: defenders run about 1.4× the risk of midfielders. |

The two accuracy columns aren't directly comparable: each model is measured in its own way. Within one table, bigger means more important.

### Recommendations

- **Expected cost of an injury:** use the **gamma GLM with severity tiers** for days missed. Its multipliers can be used directly as a pricing table.
- **The chance of a very long absence**, e.g. for policy limits: use the **lognormal** model with the same predictors. It describes the long tail more accurately.
- **Career-ending risk:** use the **logistic regression**. More complex versions did not do better.
- **Putting them together:**
  - **Formula:** expected cost ≈ chance the career ends × remaining contract + chance it doesn't × expected days out × daily salary.
  - **Example:** a 1% career-ending risk on a $50M remaining contract adds about $500,000 on its own.

### Getting a prediction for a new injury

`predict.py` takes an injury you describe and returns three things:
- the expected days out
- a realistic range: "half of similar injuries take under X days, 9 in 10 under Y"
- the chance it ends the career

For example, for a 27-year-old regular-starting defender worth €15M with a torn cruciate ligament:
- **Expected time out:** 188 days
- **Range:** half of similar injuries take under 188 days, and 9 in 10 under 319
- **Chance the injury ends his career:** about 0.5%

The same injury to a 33-year-old with typical playing time and market value comes out at 202 days and about a **5.5%** chance of ending his career. See the README for how to run it.

### Limitations

- **Soccer only.** The other sports in the database don't yet record how long injuries lasted, so the numbers may not carry over to the NFL or NBA.
- **Patterns, not causes.** For example, players with many recorded injuries tend to have *shorter* ones. That's probably because top clubs report minor knocks that lower leagues never record, not because getting injured makes players heal faster.
- **"Career-ending" is inferred.** It means the player never made another recorded professional appearance. A player who moved down to amateur football counts as career-ending, which is what an insurer of a professional career cares about.
- **Body part comes from the injury description.** It is matched by keywords, and 18% of injuries are recorded only as "unknown injury".

---

## Part 2 — How the models work, in plain terms

### What is a model here?

A model is a formula that turns facts about an injury into a prediction. Facts like what was injured, how old the player is and how much he had been playing go in; a predicted number of days comes out. The formula learns its numbers from tens of thousands of past injuries where we already know the answer.

### Training and testing

If you test a student on the exact questions they studied, you learn little about what they understand. So the players were split:
- **Training (80%):** the model learns from these.
- **Testing (20%):** the model is graded on these players, whom it never saw.

Every accuracy number in this summary comes from the test players. All of a player's injuries go on the same side of the split, so the model can't recognize someone it already studied.

### The severity model: a gamma GLM

Most injuries are short, but a few last a year or more. An ordinary average-based formula handles that lopsided pattern poorly. A **gamma GLM** (generalized linear model) is the standard insurance tool for amounts like this, which can never be negative and occasionally run very large.

Its output is easy to read because every factor works as a **multiplier**:
- Start from a baseline: an unspecified hamstring injury to a typical 26-year-old midfielder.
- A knee tear multiplies the expected days by 3.8.
- Being injured in June or July multiplies them by 1.28.
- The multipliers combine: a knee tear in June gives 3.8 × 1.28 ≈ 4.9 times the baseline.

A multiplier above 1 means longer injuries; below 1 means shorter.

### The career-ending model: logistic regression

This model predicts a **probability** between 0% and 100%. Its factors also work as multipliers, but on the **odds** of a career-ending injury rather than on days. Because these probabilities are small, multiplying the odds by 3 is very close to multiplying the risk by 3. So "a knee injury carries 8× the risk" can be read just as it sounds.

### Grouping rare injuries ("rating classes")

Each injury is placed in a group by **where** it is (knee, ankle…) and **what kind** it is (tear, fracture, bruise…). A knee *tear* and a knee *bruise* are very different, so they get separate groups.

Some combinations are rare: there might be only 40 elbow fractures. Averages from so few cases are unreliable, so rare combinations are pooled with similar ones, e.g. "fractures at other sites". Every group has at least 300 injuries behind it. Insurers do the same: they don't set a price from a handful of claims.

### Interactions

Normally each factor has one effect everywhere. An **interaction** lets a factor's effect depend on something else. For example, heavy recent playing time makes *Achilles* injuries last longer, even though it makes most injuries slightly shorter.

We tested 12 possible interactions and kept the 3 that improved accuracy on held-out data. The other 9 were left out, because adding effects that don't really exist makes a model worse on new players.

### Why several models were compared

To check that the main models are good enough, they were compared against alternatives:
- **Different shapes for the spread of injury lengths:** lognormal, inverse Gaussian and Weibull.
- **Gradient boosting:** a flexible machine-learning method that finds patterns automatically but can't produce a simple pricing table.

The gamma GLM came within about 2% of gradient boosting's accuracy while staying easy to explain. The lognormal captured the long injuries best.

### How sure are we?

Each multiplier comes with a **95% confidence interval**: a range that likely contains the true value. For common injuries the range is narrow; knee tears are 3.3× to 4.4×. For rarer groups it is wider. When two groups' ranges overlap heavily, the data can't really tell them apart.

That is why the 57 injury groups were merged into 11 **severity tiers**:
- **How:** groups next to each other in the ranking were combined whenever the difference between them could be due to chance (statistical significance at the 5% level).
- **When it stopped:** merging continued until every tier was clearly different from its neighbours.
- **No peeking:** the tiers were decided using only the training players, so the test results stay a fair check.

### Glossary

| Term | Meaning |
|---|---|
| Predictor | A fact used to make the prediction (age, body part, minutes played…). |
| Multiplier / relativity | How much a factor scales the prediction relative to the baseline; 2.0 means twice as long. |
| Odds ratio | The same idea for a yes/no outcome; at small risks, about how many times riskier. |
| Training / test set | Data the model learns from / data it is graded on. |
| Interaction | When one factor's effect depends on another. |
| Confidence interval | The range the true value probably falls in. |
| AUC | How often the model ranks a real case above a non-case; 0.5 is guessing, 1.0 is perfect. |
| Median | The middle value: half the injuries are shorter, half longer. |

---

*Details, code and charts: `severity_models.ipynb` and `career_ending_model.ipynb`. Model code: `severity.py` and `career.py`.*
