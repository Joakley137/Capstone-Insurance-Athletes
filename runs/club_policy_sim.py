"""
club_policy_sim.py — what policy would be best, and where do the real premiums sit?

    python -m runs.club_policy_sim                    # the clubs in runs/club_financials.csv (a few minutes)
    python -m runs.club_policy_sim --min-wage 8       # insure only players paid at least 8 million EUR a year

For a club-season, simulate the squad's season many times with the pricing models: each player has a negative
binomial number of injuries (frequency model); each injury is a random cell of the injury mix, its days out the
severity model's prediction times a random draw from that tier's actual/predicted spread, and it ends his career with
the career-ending model's probability. The club's loss is each player's payroll share x days out / 365, or his
share x the years of contract left if the injury ends his career. All in % of payroll. Players are independent.

A policy here is a DEDUCTIBLE: the insurer pays the days beyond d for each injury (so short injuries stay with the
club), plus the remaining contract if a career ends. The insurer prices it at (1 + loading) x its expected payout, so
its expected profit is loading x expected payout. For every deductible the club's cost is
    net cost = premium + the part of the loss the policy does not pay.
Risk is measured by CVaR: the average net cost in the worst 5% of seasons. The club's objective trades the average
against that tail, with weight kappa on the tail:
    objective = mean + kappa x (CVaR - mean)
kappa = 0 cares only about the average (buy nothing), kappa = 1 only about the tail (buy everything). Which
deductible is "best" depends on kappa and on the loading, so both are varied. With a positive loading, insurance costs
the club more on average, so a deductible is the classic way to buy protection against the large losses only.

The actual premiums cannot fix both the deductible and the loading, so the last table gives, for each deductible, the
loading the club's actual premium would contain; it shows which policies are plausible.
"""
import argparse

import numpy as np
import pandas as pd

from model import pricing
from runs import backtest
from runs import club_premiums as cp

DEDUCTIBLE_GRID = [0, 14, 28, 45, 60, 90, 120, 180, 270, 365, 10_000]    # 10,000 = no cover for days out
LOADINGS = [0.1, 0.3, 0.5]
KAPPAS = [0.25, 0.5, 0.75]
YEARS_LEFT = 2
MAX_DAYS = 1000                                                          # severity data are capped here


def cell_arrays(b, inputs):
    """For every squad player-season and injury-mix cell: predicted days (mu) and P(career-ending). Also the mix
    weights, each cell's severity tier and the tiers' actual/predicted quantile grids (rows = tiers)."""
    cells, w, n, k = cp.mix_cells(b, inputs)
    tiered = b["tiered"]
    mu = np.asarray(tiered.predict(cells)).reshape(n, k)
    cprob = np.asarray(pricing._career_p(b, cells)).reshape(n, k)
    tier_idx = np.array([tiered.tier_names.index(tiered.tier_of[c]) for c in cells["injury_class"].iloc[:k].astype(str)])
    ratios = np.vstack([tiered.ratio_quantiles[t] for t in tiered.tier_names])
    return mu, cprob, w, tier_idx, ratios


def simulate_injuries(share, lam, alpha, mu, cprob, w, tier_idx, ratios, n_sims, rng, max_days=MAX_DAYS, insured=None):
    """Every injury in n_sims simulated seasons: which simulation, the payroll share of the player, days out, whether
    it ended his career and whether the player is insured. share, lam: (players,); mu, cprob: (players, cells);
    alpha: NB dispersion (0 = Poisson); insured: (players,) booleans, default everyone."""
    n, k = mu.shape
    mean = np.broadcast_to(lam, (n_sims, n))
    counts = rng.poisson(rng.gamma(1 / alpha, alpha * mean) if alpha > 0 else mean)
    sim = np.repeat(np.arange(n_sims), counts.sum(axis=1))
    pl = np.repeat(np.tile(np.arange(n), n_sims), counts.ravel())
    j = rng.choice(k, size=len(pl), p=w / w.sum())
    g = rng.integers(0, ratios.shape[1], len(pl))
    days = np.minimum(mu[pl, j] * ratios[tier_idx[j], g], max_days)
    ended = rng.random(len(pl)) < cprob[pl, j]
    ok = np.ones(n, bool) if insured is None else insured
    return sim, share[pl], days, ended, ok[pl]


def outcomes(sim, share_k, days, ended, insured_k, n_sims, deductibles=DEDUCTIBLE_GRID, loadings=LOADINGS,
             years_left=YEARS_LEFT, alpha=0.95):
    """Per deductible and loading: expected payout, the club's mean and CVaR of net cost, and the insurer's profit
    (all % of payroll). Net cost = premium + loss - payout."""
    total = lambda x: np.bincount(sim, weights=x, minlength=n_sims) * 100
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


def club_outcomes(rows, mu, cprob, w, tier_idx, ratios, alpha, club, n_sims=3000, seed=0, payroll=None, min_wage=0.0,
                  **kw):
    """Average the outcomes over a club's seasons. rows: cp.squad_rows() (share, lam, club, season), aligned with mu
    and cprob. With min_wage (EUR m a year) and payroll ((club, season) -> EUR m), only players paid at least that
    much are insured; the club still bears the loss from the others."""
    rng = np.random.default_rng(seed)
    out = []
    for (c, season), g in rows.groupby(["club", "season"]):
        if c != club:
            continue
        ix = g.index.to_numpy()
        insured = None
        if min_wage > 0:
            insured = (g["share"] * payroll[(c, season)]).to_numpy() >= min_wage
        sim, sh, days, ended, ins = simulate_injuries(g["share"].to_numpy(), g["lam"].to_numpy(), alpha, mu[ix],
                                                      cprob[ix], w, tier_idx, ratios, n_sims, rng, insured=insured)
        out.append(outcomes(sim, sh, days, ended, ins, n_sims, **kw))
    return pd.concat(out).groupby(["deductible", "loading"], as_index=False).mean()


def main():
    ap = argparse.ArgumentParser(description="Simulate deductible policies for the clubs in club_financials.csv")
    ap.add_argument("--min-wage", type=float, default=0.0,
                    help="insure only players paid at least this many EUR million a year (see club_premiums for a fitted value)")
    min_wage = ap.parse_args().min_wage
    fin, fd, squad, b = cp.load_base()
    payroll = {(r.club, r.season): r.payroll for r in fin.itertuples()}
    rows = cp.squad_rows(squad, fd, b)
    mu, cprob, w, tier_idx, ratios = cell_arrays(b, backtest.season_inputs(rows))
    alpha = b["frequency"].alpha
    prem = fin.groupby("club")["premium_pct"].mean()
    pd.set_option("display.width", 200)
    for club in sorted(rows["club"].unique()):
        o = club_outcomes(rows, mu, cprob, w, tier_idx, ratios, alpha, club, payroll=payroll, min_wage=min_wage)
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
