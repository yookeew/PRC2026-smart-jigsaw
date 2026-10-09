"""Day-sum drift model (reports/v57_daysum_model_preregistration.md, C1/C2/C3).

Per airport over a period's days: lam_g from the expectation solve (src/post/pru_recon2.py), then
  lam_g = mu_{a,m} + sum_k beta_{a,k} w_{g,k} + u_g       (OLS, month fixed effects)
with w_{g,k} the share of the day's sensitivity sum P_eta(d_hat + eta > 0) (at lam = 0) in group k, and an
empirical-Bayes shrink of u_g: a_a = max(0, 1 - s2_a / Var(u_hat)). Row shift (d units):
  delta_i = mu_hat + beta_{k(i)} + a_a u_hat_g ;  new taxi = max(0, p - delta).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from post import pru_recon2 as R

FAMILIES = {
    "F1": ["band15_60", "band60"],
    "F2": ["nmu"],
    "F3": ["obs"],
    "F4": ["tod9_15", "tod15_24"],
}


def indicators(x: pl.DataFrame) -> dict[str, np.ndarray]:
    """Non-reference group indicators per row (label-free)."""
    dh = x["ts"].to_numpy().astype(float) - x["p"].to_numpy().astype(float)
    hour = x["hour"].to_numpy()
    return {"band15_60": (dh >= 900) & (dh < 3600), "band60": dh >= 3600,
            "nmu": x["nmu"].to_numpy().astype(bool), "obs": x["obs"].to_numpy().astype(bool),
            "tod9_15": (hour >= 9) & (hour < 15), "tod15_24": hour >= 15}


def day_table(x: pl.DataFrame, dists_by_apt: dict, airports, row_dist=None) -> list[dict]:
    """One entry per solved airport-day of the given airports: g (row idx), apt, per (month), lam (expectation solve,
    uniform), w (dict of sensitivity shares at lam = 0), n. row_dist: optional per-row callable returning
    (idx, dists, fixed_contribution, solve_mask) for special airports (LIRF); default = all rows, tracked split."""
    p = x["p"].to_numpy().astype(float)
    dh = x["ts"].to_numpy().astype(float) - p
    apt, per = x["apt"].to_numpy(), x["mo"].to_numpy()
    trk = x["trk"].to_numpy().astype(bool)
    Dv = x["D"].to_numpy().astype(float)
    ind = indicators(x)
    out = []
    for g in R.groups_of(x):
        a = apt[g[0]]
        if a not in airports:
            continue
        if row_dist is not None:
            idx, dists, fixed, move = row_dist(g)
        else:
            dists, idx, fixed, move = list(dists_by_apt[a]), trk[g].astype(int), 0.0, np.ones(len(g), bool)
        target = Dv[g[0]] * 60.0 - fixed
        gm = g[move]
        lam = R.solve(dh[gm], np.ones(len(gm)), idx[move], dists, target)
        if not np.isfinite(lam):
            continue
        P = np.zeros(len(gm))
        for j, dist in enumerate(dists):
            k = idx[move] == j
            if k.any():
                P[k] = dist.var_p(dh[gm][k])[1]
        tot = max(P.sum(), 1e-9)
        w = {c: float(P[ind[c][gm]].sum() / tot) for c in ind}
        out.append({"apt": a, "per": per[g[0]], "g": g, "gm": gm, "lam": lam, "w": w, "n": len(g)})
    return out


def design(days: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray, list]:
    pers = sorted(set(d["per"] for d in days))
    X = np.column_stack([np.array([d["per"] == m for d in days], float) for m in pers]
                        + [np.array([d["w"][c] for d in days]) for c in cols]) if days else np.zeros((0, 0))
    y = np.array([d["lam"] for d in days])
    return X, y, pers


def fit(days: list[dict], cols: list[str]):
    X, y, pers = design(days, cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu = dict(zip(pers, coef[:len(pers)]))
    beta = dict(zip(cols, coef[len(pers):]))
    u = y - X @ coef
    return mu, beta, u


def loo_rmse(days: list[dict], cols: list[str]) -> tuple[float, int]:
    """Leave-one-day-out RMSE of lam (seconds) for one airport's days."""
    X, y, _ = design(days, cols)
    err = []
    for i in range(len(y)):
        k = np.ones(len(y), bool); k[i] = False
        if np.linalg.matrix_rank(X[k]) < X.shape[1]:
            continue
        coef, *_ = np.linalg.lstsq(X[k], y[k], rcond=None)
        err.append(y[i] - X[i] @ coef)
    err = np.array(err)
    return float(np.sqrt(np.mean(err ** 2))), len(err)


def shifts(x: pl.DataFrame, days_by_apt: dict, cols_by_apt: dict, s2_by_apt: dict, level: bool = True):
    """Row shifts (d units) for the solved days. level=False removes the month level (C3: day variation only).
    Returns delta (zeros elsewhere) and a per-airport summary."""
    delta = np.zeros(x.height)
    ind = indicators(x)
    summ = {}
    for a, days in days_by_apt.items():
        if not days:
            continue
        cols = cols_by_apt.get(a, [])
        mu, beta, u = fit(days, cols)
        dof = max(len(u) - len(mu) - len(cols), 1)
        vu = float(np.sum(u ** 2) / dof)
        aa = max(0.0, 1.0 - s2_by_apt[a] / vu) if vu > 0 else 0.0
        for d, ug in zip(days, u):
            gm = d["gm"]
            base = (mu[d["per"]] if level else 0.0) + aa * ug
            dd = np.full(len(gm), base)
            for c in cols:
                dd = dd + beta[c] * ind[c][gm] * (1.0 if level else 0.0)
            if not level:      # day variation only: the day's deviation from the month level, incl. its mix part
                dd = np.full(len(gm), aa * ug + sum(beta[c] * (d["w"][c] - np.mean([e["w"][c] for e in days if e["per"] == d["per"]]))
                                                    for c in cols))
            delta[gm] = dd
        summ[a] = {"a": round(aa, 3), "var_u": round(vu, 1), "s2": round(s2_by_apt[a], 1),
                   "mu": {k: round(v, 1) for k, v in mu.items()}, "beta": {k: round(v, 1) for k, v in beta.items()}}
    return delta, summ


def noise_s2(x: pl.DataFrame, days: list[dict]) -> pl.DataFrame:
    """Per airport-day lam - delta* (delta* = mean(p - y) over the day's rows) with its (airport, per) mean removed."""
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy().astype(float)
    t = pl.DataFrame({"apt": [d["apt"] for d in days], "per": [d["per"] for d in days],
                      "e": [d["lam"] - float(np.mean(p[d["g"]] - y[d["g"]])) for d in days]})
    return t.with_columns(ec=pl.col("e") - pl.col("e").mean().over("apt", "per"))
