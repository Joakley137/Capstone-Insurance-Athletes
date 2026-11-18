"""
premium_calculator.py - a super basic season premium calculator for one soccer player.

Uses the coefficients from MODEL_EQUATIONS.md in Capstone-Insurance-Athletes.
No packages needed - just Python.

    python3 premium_calculator.py

To price your own player, change the numbers at the bottom of this file.
"""
import math

ALPHA = 0.223  # negative binomial dispersion from the frequency model

# Coefficients copied from MODEL_EQUATIONS.md (Premier League midfielder = reference, so 0)
FREQ_LEAGUE = {"Premier League": 0, "LaLiga": -0.0279, "Serie A": 0.2922, "Bundesliga": 0.3251, "Ligue 1": -0.1235}
FREQ_POSITION = {"Midfielder": 0, "Goalkeeper": -0.4278, "Defender": 0.0468, "Attacker": 0.0898}
SEV_LEAGUE = {"Premier League": 0, "LaLiga": -0.1756, "Serie A": -0.3752, "Bundesliga": -0.3172, "Ligue 1": -0.1362}
SEV_POSITION = {"Midfielder": 0, "Attacker": -0.0143, "Defender": 0.0371, "Goalkeeper": 0.0937}
CAREER_POSITION = {"Midfielder": 0, "Attacker": -0.2853, "Defender": 0.2919, "Goalkeeper": 0.1437}

# Typical player's deferment curve from MODEL_SUMMARY.md (25yo Serie A defender, 1,898 min, 2 prior injuries)
DEFERMENTS = [30, 60, 90, 180]
P_LONGER = [0.264, 0.112, 0.062, 0.019]   # P(an injury lasts longer than the deferment)
AVG_DAYS = [77, 125, 167, 264]            # average length of an injury that does


# ---------------------------------------------------------------------------
# Step 1: expected injuries this season (frequency model - exact)
# ---------------------------------------------------------------------------
def expected_injuries(age, position, league, minutes_last_season, prior_injuries, days_out_last_season):
    a = age - 26
    games = minutes_last_season / 90
    no_minutes = 1 if minutes_last_season == 0 else 0
    log_rate = (-1.4086 + FREQ_LEAGUE[league] + FREQ_POSITION[position]
                - 0.0073 * a - 0.0037 * a ** 2
                + 0.2098 * math.log(1 + games) - 0.0202 * no_minutes
                + 0.4127 * math.log(1 + prior_injuries) + 0.0778 * math.log(1 + days_out_last_season))
    return math.exp(log_rate)


# ---------------------------------------------------------------------------
# Step 2: chance an injury outlasts the deferment (severity model - approximate)
# ---------------------------------------------------------------------------
def _severity_terms(age, position, league, prior_injuries):
    a = age - 26
    return (SEV_LEAGUE[league] + SEV_POSITION[position]
            + 0.0028 * a + 0.0009 * a ** 2 - 0.1144 * math.log(1 + prior_injuries))


def _log_interp(x, xs, ys):
    """Straight line between table points on a log-log scale (extends past the ends)."""
    i = 0
    while i < len(xs) - 2 and x > xs[i + 1]:
        i += 1
    t = (math.log(x) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
    return math.exp(math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i])))


def injury_length(age, position, league, prior_injuries, deferment):
    """Returns (q, avg_days): q = P(injury > deferment), avg_days = average length of those injuries.
    A player whose injuries run 1.4x longer than the typical player looks like the typical
    player with a deferment 1.4x shorter."""
    typical = _severity_terms(25, "Defender", "Serie A", 2)
    multiplier = math.exp(_severity_terms(age, position, league, prior_injuries) - typical)
    q = min(1.0, _log_interp(deferment / multiplier, DEFERMENTS, P_LONGER))
    avg_days = multiplier * _log_interp(deferment / multiplier, DEFERMENTS, AVG_DAYS)
    return q, avg_days


# ---------------------------------------------------------------------------
# Step 3: chance an injury ends the career (career model - approximate)
# ---------------------------------------------------------------------------
def _career_terms(age, position, minutes_last_season, career_minutes, prior_injuries):
    a = age - 26
    no_minutes = 1 if minutes_last_season == 0 else 0
    return (CAREER_POSITION[position] + 0.2539 * a - 0.0040 * a ** 2
            - 0.2955 * math.log(1 + prior_injuries)
            - 0.6049 * math.log(1 + minutes_last_season / 90) - 0.4040 * no_minutes
            - 0.2481 * math.log(1 + career_minutes / 90))


def career_ending_chance(age, position, minutes_last_season, career_minutes, prior_injuries):
    """Scales the doc's example (27yo defender, c = 0.25%) by the logistic model's odds ratio."""
    example = _career_terms(27, "Defender", 2500, 15000, 4)
    odds = 0.0025 * math.exp(_career_terms(age, position, minutes_last_season, career_minutes, prior_injuries) - example)
    return odds / (1 + odds)


# ---------------------------------------------------------------------------
# Step 4: put it together
# ---------------------------------------------------------------------------
def p_at_least_one(mean):
    """Negative binomial chance of at least one event, as in pricing.combine()."""
    return 1 - (1 + ALPHA * mean) ** (-1 / ALPHA)


def premium(age, position="Defender", league="Premier League", minutes_last_season=2500,
            career_minutes=15000, prior_injuries=4, days_out_last_season=0,
            deferment=60, annual_wage=5_000_000, percent_covered=1.0, sum_insured=20_000_000, loading=0.30):
    lam = expected_injuries(age, position, league, minutes_last_season, prior_injuries, days_out_last_season)
    q, avg_days = injury_length(age, position, league, prior_injuries, deferment)
    c = career_ending_chance(age, position, minutes_last_season, career_minutes, prior_injuries)

    days_paid = max(0, avg_days - deferment)                     # policy pays after the deferment
    daily_benefit = percent_covered * annual_wage / 365          # share of the wage the policy pays per day
    wage_cost = lam * q * days_paid * daily_benefit              # expected wages paid for long injuries
    career_cost = lam * c * sum_insured                          # expected career-ending payout
    total = (wage_cost + career_cost) * (1 + loading)

    return {
        "expected injuries": lam,
        "P(any injury)": p_at_least_one(lam),
        "P(covered injury)": p_at_least_one(lam * q),
        "P(career-ending)": p_at_least_one(lam * c),
        "expected wage cost": wage_cost,
        "expected career-ending cost": career_cost,
        "premium": total,
    }


def show(result):
    for name, value in result.items():
        if name.startswith("P("):
            print(f"  {name:<28} {value:.2%}")
        elif "cost" in name or name == "premium":
            print(f"  {name:<28} EUR {value:,.0f}")
        else:
            print(f"  {name:<28} {value:.3f}")


if __name__ == "__main__":
    # ---- Change these numbers to price a different player ----
    result = premium(
        age=27,
        position="Defender",          # Midfielder, Defender, Attacker, Goalkeeper
        league="Premier League",      # Premier League, LaLiga, Serie A, Bundesliga, Ligue 1
        minutes_last_season=2500,
        career_minutes=15000,
        prior_injuries=4,
        days_out_last_season=0,
        deferment=60,                 # days before the policy starts paying
        annual_wage=5_000_000,        # EUR
        percent_covered=0.80,         # policy pays 80% of the wage while the player is out
        sum_insured=20_000_000,       # EUR paid if the career ends
        loading=0.30,                 # 30% on top of expected cost
    )
    print("27-year-old Premier League defender:")
    show(result)
