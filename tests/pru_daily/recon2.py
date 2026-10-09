exec(open(os.environ["SCR"] + "/recon.py").read().split("print(f\"rows")[0])
lam_const = {}
def solve(w):
    out = np.full(len(y), np.nan)
    for m in groups:
        Dk = Dv[m[0]]
        if not np.isfinite(Dk):
            continue
        ww = w[m]
        if ww.sum() == 0: continue
        f = lambda lam: np.maximum(0.0, dh[m] + lam * ww).sum() / 60.0 - Dk
        if f(-5e4) * f(5e4) > 0: continue
        out[m] = brentq(f, -5e4, 5e4, xtol=0.01)
    return out
L1 = solve(np.ones(len(y)))
L2 = solve((dh > -600).astype(float))
L3 = solve((dh > -600).astype(float) * (dh < 3 * 3600))
for nm_, L, wv in (("all", L1, np.ones(len(y))), ("dpos", L2, (dh > -600).astype(float)), ("dpos_body", L3, ((dh > -600) & (dh < 10800)).astype(float))):
    for a in (0.5, 0.75, 1.0):
        for cap in (300, 900, 1e9):
            sh = np.nan_to_num(np.clip(a * L, -cap, cap))
            for app in ("w", "all"):
                applied = sh * (wv if app == "w" else 1.0)
                q = np.maximum(ts - (dh + applied), 0.0)
                if (a, cap) in ((1.0, 1e9), (0.75, 1e9), (0.5, 1e9), (1.0, 900), (0.75, 900), (0.75, 300)):
                    rep(f"{nm_}/{app} a{a} c{int(min(cap,9999))}", q)
