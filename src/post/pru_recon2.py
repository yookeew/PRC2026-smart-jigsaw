"""Daily reconciliation v2 with EUROCONTROL's all-causes pre-departure delay (reports/v57_pru_recon2_preregistration.md).

Per airport-day g (rows assigned by predicted off-block date, as src/post/pru_daily_recon.py) the published
D_g = sum max(0, AOBT - SOBT) [min]. With d = (T - S) - taxi and residual d_i = d_hat_i + delta_g + eta_i:

  R0  point solve   sum max(0, d_hat_i + lam) = 60 D                  (v55; apply 0.75 * lam)
  R1  expectation   sum G(d_hat_i + lam w_i) = 60 D, G(x) = E_eta max(0, x + eta)   (apply a * lam * w_i)
  R2  R1 with an empirical-Bayes shrink per day: a_g = tau2 / (tau2 + sig2_g),
      sig2_g = sum Var_eta max(0, d_hat_i + lam w_i + eta) / (sum w_i P(d_hat_i + lam w_i + eta > 0))^2,
      tau2 = max(mean_g lam_g^2 - mean_g sig2_g, 25) over the target period's days of the airport (label-free)
  weights w_i: 1 (uniform), w_trk for rows in an ADS-B pushback tier (R3), p_i / mean_g p (R4).

eta samples (d-residual minus its airport-day mean) come from labelled calibration data, per airport and per
tracked / untracked status. Usage: tab = solve_days(x, dists, kind); q = apply(x, tab, a) with a a number or "eb".
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import brentq

TRACKED = ("dwell", "pass", "fb_dwell")


class EtaDist:
    """Empirical eta sample: G(x) = E max(0, x + eta), Var and P(x + eta > 0), exact, vectorised."""

    def __init__(self, eta: np.ndarray, max_n: int = 20001):
        e = np.sort(np.asarray(eta, dtype=np.float64))
        if len(e) > max_n:                          # quantile thinning keeps the tails
            e = np.quantile(e, np.linspace(0, 1, max_n))
        self.e, self.n = e, len(e)
        self.c1 = np.concatenate([[0.0], np.cumsum(e)])
        self.c2 = np.concatenate([[0.0], np.cumsum(e * e)])

    def _tail(self, x):
        k = np.searchsorted(self.e, -x, side="right")           # eta_j <= -x contribute 0
        return self.n - k, self.c1[-1] - self.c1[k], self.c2[-1] - self.c2[k]

    def g(self, x):
        m, s1, _ = self._tail(x)
        return (m * x + s1) / self.n

    def var_p(self, x):
        m, s1, s2 = self._tail(x)
        g = (m * x + s1) / self.n
        e2 = (m * x * x + 2 * x * s1 + s2) / self.n
        return np.maximum(e2 - g * g, 0.0), m / self.n


class PointDist:
    """Degenerate eta = 0 (R0's point solve)."""

    def g(self, x):
        return np.maximum(0.0, x)

    def var_p(self, x):
        return np.zeros_like(x), (x > 0).astype(float)


def eta_sample(y, p, key) -> np.ndarray:
    """d-residual (p - y) minus its mean within key (airport-day)."""
    r = np.asarray(p, float) - np.asarray(y, float)
    _, inv = np.unique(key, return_inverse=True)
    mu = np.bincount(inv, r) / np.bincount(inv)
    return r - mu[inv]


def _gsum(x, idx, dists):
    tot = 0.0
    for j, dist in enumerate(dists):
        k = idx == j
        if k.any():
            tot += dist.g(x[k]).sum()
    return tot


def solve(dh, w, idx, dists, target):
    """lam with sum_i G_{idx_i}(dh_i + lam w_i) = target; nan if not bracketed."""
    f = lambda lam: _gsum(dh + lam * w, idx, dists) - target  # noqa: E731
    lo, hi = -5e4, 5e4
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=0.01)


def noise_var(dh, w, idx, dists, lam):
    x = dh + lam * w
    v = np.zeros(len(x)); pp = np.zeros(len(x))
    for j, dist in enumerate(dists):
        k = idx == j
        if k.any():
            v[k], pp[k] = dist.var_p(x[k])
    den = (w * pp).sum()
    return v.sum() / den ** 2 if den > 0 else np.inf


def groups_of(x: pl.DataFrame):
    """Row-index groups per airport-day whose row count is within +-5% of the published FLT_DEP_IFR_2."""
    key = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
    D, nD = x["D"].to_numpy().astype(float), x["nD"].to_numpy().astype(float)
    cnt = np.bincount(inv)[inv]
    ok = np.isfinite(nD) & np.isfinite(D) & (np.abs(cnt / np.where(nD > 0, nD, np.nan) - 1) <= 0.05)
    return [g for g in groups if ok[g[0]]]


def weights(x: pl.DataFrame, g, kind: str, w_trk: float = 1.0):
    if kind == "trk":
        return np.where(x["trk"].to_numpy()[g], w_trk, 1.0)
    if kind in ("mult", "trkmult"):
        p = x["p"].to_numpy()[g].astype(float)
        w = p / max(p.mean(), 1.0)
        if kind == "trkmult":
            w = w * np.where(x["trk"].to_numpy()[g], w_trk, 1.0)
        return w
    return np.ones(len(g))


def solve_days(x: pl.DataFrame, dists_by_apt: dict | None, kind: str = "uniform", w_trk: float = 1.0,
               skip=("LIRF", "LSZH"), period_col: str = "mo") -> pl.DataFrame:
    """x: rows with p, ts, apt, day, D, nD, [trk], period_col. dists_by_apt None -> point solve (R0); else
    dists_by_apt[apt] = (untracked EtaDist, tracked EtaDist). One row per solved airport-day: g (row indices), lam, s2."""
    p = x["p"].to_numpy().astype(float)
    dh = x["ts"].to_numpy().astype(float) - p
    apt, per = x["apt"].to_numpy(), x[period_col].to_numpy()
    trk = x["trk"].to_numpy().astype(bool) if "trk" in x.columns else np.zeros(len(p), bool)
    Dv = x["D"].to_numpy().astype(float)
    out = []
    for g in groups_of(x):
        k = apt[g[0]]
        if k in skip:
            continue
        if dists_by_apt is None:
            dists, idx = [PointDist()], np.zeros(len(g), int)
        else:
            dists, idx = list(dists_by_apt[k]), trk[g].astype(int)
        w = weights(x, g, kind, w_trk)
        lam = solve(dh[g], w, idx, dists, Dv[g[0]] * 60.0)
        if not np.isfinite(lam):
            continue
        out.append({"apt": k, "per": per[g[0]], "g": g, "w": w, "lam": lam, "s2": noise_var(dh[g], w, idx, dists, lam)})
    return out


def apply(x: pl.DataFrame, days: list, a) -> np.ndarray:
    """New taxi: d_hat + a_g * lam * w on each solved day; a a number or 'eb' (R2's per-day shrink)."""
    p = x["p"].to_numpy().astype(float)
    ts = x["ts"].to_numpy().astype(float)
    q = p.copy()
    if not days:
        return q
    if a == "eb":
        t = pl.DataFrame({"apt": [d["apt"] for d in days], "per": [d["per"] for d in days],
                          "lam": [d["lam"] for d in days], "s2": [d["s2"] for d in days]})
        t = t.with_columns(tau2=((pl.col("lam") ** 2).mean() - pl.col("s2").mean()).over("apt", "per").clip(25.0, None))
        av = (t["tau2"] / (t["tau2"] + t["s2"])).to_numpy()
    else:
        av = np.full(len(days), float(a))
    for d, ag in zip(days, av):
        g = d["g"]
        q[g] = np.maximum(ts[g] - (ts[g] - p[g] + ag * d["lam"] * d["w"]), 0.0)
    return q


def shrink_table(days: list) -> pl.DataFrame:
    t = pl.DataFrame({"apt": [d["apt"] for d in days], "per": [d["per"] for d in days],
                      "lam": [d["lam"] for d in days], "s2": [d["s2"] for d in days]})
    t = t.with_columns(tau2=((pl.col("lam") ** 2).mean() - pl.col("s2").mean()).over("apt", "per").clip(25.0, None))
    return t.with_columns(a=pl.col("tau2") / (pl.col("tau2") + pl.col("s2")))
