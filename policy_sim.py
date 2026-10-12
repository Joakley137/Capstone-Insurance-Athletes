"""
policy_sim.py — what policy would be best, and where do the real premiums sit?

    python policy_sim.py                      # the four clubs in club_financials.csv
    python policy_sim.py --min-wage 8         # insure only players paid at least 8 million EUR a year

For a club-season, simulate the squad's season many times: each player has a Poisson number of injuries (frequency
model), each injury a number of days out (lognormal severity model, for a randomly drawn real injury) and a chance of
ending his career (career-ending model). The club's loss is each player's payroll share x days out / 365, or his
share x the years of contract left if the injury ends his career. All in % of payroll. Players are independent.

A policy here is a DEDUCTIBLE: the insurer pays the days beyond d for each injury (so short injuries stay with the
club), plus the remaining contract if a career ends. The insurer prices it at (1 + loading) x its expected payout, so
its expected profit is loading x expected payout. For every deductible the club's cost is
    net cost = premium + the part of the loss the policy does not pay.
Risk is measured by CVaR: the average net cost in the worst 5% of seasons. The club's objective trades off the
average against that tail, with weight kappa on the tail:
    objective = mean + kappa x (CVaR - mean)
kappa = 0 cares only about the average (buy nothing), kappa = 1 only about the tail (buy everything). Which
deductible is "best" depends on kappa and on the loading, so both are varied. With a positive loading, insurance costs
the club more on average, so a deductible is the classic way to buy protection against the large losses only.

The actual premiums cannot fix both the deductible and the loading, so the last table gives, for each deductible,
the loading the club's actual premium would contain; it lets you see which policies are plausible.

Needs the same tables as premium_test.py.
"""
import numpy as np
import pandas as pd

import career
import frequency as freq
import premium_test as pt
import severity as sev

DEDUCTIBLE_GRID = [0, 14, 28, 45, 60, 90, 120, 180, 270, 365, 10_000]    # 10,000 = no cover for days out
LOADINGS = [0.1, 0.3, 0.5]
KAPPAS = [0.25, 0.5, 0.75]
YEARS_LEFT = 2
MAX_DAYS = 1000                                                          # severity data are capped here


def simulate_injuries(share, lam, mu, pce, sigma, n_sims, rng, max_days=MAX_DAYS, insured=None):
    """Every injury in n_sims simulated seasons: which simulation, the payroll share of the player, days out and
    whether it ended his career, and whether the player is insured. share, lam: (players,); mu, pce: (players, mix);
    insured: (players,) booleans, default everyone."""
    n, m = mu.shape
    counts = rng.poisson(lam, size=(n_sims, n))
    sim = np.repeat(np.arange(n_sims), counts.sum(axis=1))
    pl = np.repeat(np.tile(np.arange(n), n_sims), counts.ravel())
    j = rng.integers(0, m, len(pl))
    days = np.minimum(np.exp(mu[pl, j] + sigma * rng.standard_normal(len(pl))), max_days)
    ended = rng.random(len(pl)) < pce[pl, j]
    ok = np.ones(n, bool) if insured is None else insured
    return sim, share[pl], days, ended, ok[pl]


def outcomes(sim, share_k, days, ended, insured_k, n_sims, deductibles=DEDUCTIBLE_GRID, loadings=LOADINGS, years_left=YEARS_LEFT,
             alpha=0.95):
    """Per deductible and loading: expected payout, the club's mean and CVaR of net cost, and the insurer's profit
    (all % of payroll). Net cost = premium + loss - payout."""
    total = lambda w: np.bincount(sim, weights=w, minlength=n_sims) * 100
    loss = total(share_k * np.where(ended, years_left, days / 365))
    tail = int(np.ceil((1 - alpha) * n_sims))
    rows = []
    for d in deductibles:
        payout = total(insured_k * share_k * np.where(ended, years_left, np.maximum(days - d, 0) / 365))
        for theta in loadings:
            premium = (1 + theta) * payout.mean()
            net = premium + loss - payout
            rows.append({"deductible": d, "loading": theta, "premium": premium, "payout": payout.mean(),
                         "insurer_profit": premium - payout.mean(), "club_mean": net.mean(),
                         "club_cvar": np.sort(net)[-tail:].mean(), "club_sd": net.std()})
    return pd.DataFrame(rows)


def best_deductible(o, kappas=KAPPAS):
    """The deductible that minimises mean + kappa x (CVaR - mean), per loading and kappa."""
    o = o.assign(**{f"obj_{k}": o["club_mean"] + k * (o["club_cvar"] - o["club_mean"]) for k in kappas})
    rows = []
    for theta, g in o.groupby("loading"):
        r = {"loading": theta}
        for k in kappas:
            r[f"kappa={k}"] = g.loc[g[f"obj_{k}"].idxmin(), "deductible"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("loading")


def club_outcomes(p, mu, pce, sigma, club, n_sims=3000, seed=0, payroll=None, min_wage=0.0, **kw):
    """Average the outcomes over a club's seasons. With min_wage (EUR m a year) and payroll (club, wy -> EUR m),
    only players paid at least that much are insured; the club still bears the loss from the others."""
    rng = np.random.default_rng(seed)
    out = []
    for (c, wy), g in p.groupby(["club", "wy"]):
        if c != club:
            continue
        ix = g.index.to_numpy()
        insured = None
        if min_wage > 0:
            insured = (g["share"] * payroll[(c, wy)]).to_numpy() >= min_wage
        sim, sh, days, ended, ins = simulate_injuries(g["share"].to_numpy(), g["lam"].to_numpy(), mu[ix], pce[ix],
                                                      sigma, n_sims, rng, insured=insured)
        out.append(outcomes(sim, sh, days, ended, ins, n_sims, **kw))
    return pd.concat(out).groupby(["deductible", "loading"], as_index=False).mean()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Simulate deductible policies for the clubs in club_financials.csv")
    ap.add_argument("--min-wage", type=float, default=0.0,
                    help="insure only players paid at least this many EUR million a year (see premium_test.py for a fitted value)")
    min_wage = ap.parse_args().min_wage
    fin = pt.load_financials()
    payroll = {(r.club, r.wy): r.payroll for r in fin.itertuples()}
    last = career.last_complete_season_start(sev.player_seasons("Soccer")).year
    squad = pt.load_squad(last)
    p, mu, pce, sigma = pt.fit_squad_inputs(squad)
    prem = fin[fin["wy"].between(freq.DEFAULT_START_YEAR, last)].groupby("club")["premium_pct"].mean()
    pd.set_option("display.width", 200)
    for club in sorted(p["club"].unique()):
        o = club_outcomes(p, mu, pce, sigma, club, payroll=payroll, min_wage=min_wage)
        print(f"\n=== {club}: average premium {prem.get(club, float('nan')):.2f}% of payroll ===")
        view = o[o["loading"] == 0.3].set_index("deductible")[["payout", "premium", "insurer_profit", "club_mean", "club_cvar"]]
        print("Outcomes at a 30% loading, % of payroll (deductible 10000 = no cover):")
        print(view.round(2).to_string())
        print("Best deductible (days) by loading and tail weight kappa:")
        print(best_deductible(o).to_string())
        fair = o[o["loading"] == 0.1].set_index("deductible")["payout"]
        print("Loading the actual premium would contain at each deductible:")
        print((prem.get(club, np.nan) / fair - 1).mul(100).round(0).rename(lambda d: f"{d}d").to_frame("loading %").T.to_string())


if __name__ == "__main__":
    main()
