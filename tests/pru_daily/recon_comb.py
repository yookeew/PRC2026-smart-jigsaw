exec(open(os.environ["SCR"] + "/recon2.py").read().split("for nm_, L, wv")[0])
lq = np.load(os.environ["SCR"] + "/lirf_bayes_q.npy")
lmap = dict(zip(lq[:, 0].astype(np.int64), lq[:, 1]))
ids = d["MVT_ID_mvt"].to_numpy()
sh = np.nan_to_num(0.75 * L1)
q = np.maximum(ts - (dh + sh), 0.0)
isl = apt == "LIRF"
q_comb = q.copy(); q_comb[isl] = np.array([lmap[i] for i in ids[isl]])
q_nl = p.copy(); q_nl[~isl] = q[~isl]
rep("shift non-LIRF only", q_nl)
rep("shift non-LIRF + LIRF bayes", q_comb)
q_both = q_comb.copy()
# LIRF: Bayes, then the day shift recomputed on top for LIRF too
rep("shift all incl LIRF (ref)", q)
# LIRF non-hedged rows: day shift solved on top of the Bayes-updated predictions, applied to non-hedged rows only
TAU = 1100
Hl = isl & (ts >= 2700) & (p > TAU + 60) & (p < ts - 60)
dh2 = ts - q_comb
q3 = q_comb.copy()
for m in groups:
    if apt[m[0]] != "LIRF" or not np.isfinite(Dv[m[0]]): continue
    wv = (~Hl[m]).astype(float)
    f = lambda lam: np.maximum(0.0, dh2[m] + lam * wv).sum() / 60.0 - Dv[m[0]]
    if f(-5e4) * f(5e4) > 0: continue
    lam = brentq(f, -5e4, 5e4, xtol=0.01)
    for a in (0.5,):
        q3[m] = np.maximum(ts[m] - (dh2[m] + a * lam * wv), 0)
rep("comb + LIRF nonhedged shift a0.5", q3)
