exec(open(os.environ["SCR"] + "/recon2.py").read().split("for nm_, L, wv")[0])
import collections
# holdout lambda medians per airport x month
dfl = pl.DataFrame({"apt": apt, "mo": mo, "L": L1, "key": key}).unique("key")
print(dfl.group_by("apt", "mo").agg(lam_med=pl.col("L").median()).pivot(on="mo", index="apt", values="lam_med").sort("apt"))
# Jensen/structural offset J per airport, cross-month: J = median over the other month's days of (D - sum max(0, d_true_hat?))
# use lambda-space: subtract the other month's median lambda per airport
med = {(a, m): float(np.nanmedian(L1[(apt == a) & (mo == m)])) for a in set(apt) for m in (1, 7)}
for a_ in (0.75, 1.0):
    Lc = np.array([L1[i] - med[(apt[i], 8 - mo[i])] for i in range(len(y))])
    q = np.maximum(ts - (dh + np.nan_to_num(a_ * Lc)), 0.0)
    qq = p.copy(); qq[apt != "LIRF"] = q[apt != "LIRF"]
    rep(f"centred a{a_} nonLIRF", qq)
q = np.maximum(ts - (dh + np.nan_to_num(0.75 * L1)), 0.0); qq = p.copy(); qq[apt != "LIRF"] = q[apt != "LIRF"]
rep("raw a0.75 nonLIRF", qq)
