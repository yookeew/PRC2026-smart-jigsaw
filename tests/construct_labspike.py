import polars as pl, glob, numpy as np
d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed")
for ph in ("DEP","ARR"):
    z = d.filter(pl.col("PHASE_mvt")==ph)
    key = "ADEP_mvt" if ph=="DEP" else "ADES_mvt"
    for apt in sorted(z[key].unique().to_list()):
        y = z.filter(pl.col(key)==apt)["TAXITIME_SEC_mvt"].drop_nulls().to_numpy()
        y = y[(y>=0)&(y<7200)]
        h = np.bincount(y, minlength=7200).astype(float)
        # smooth by +-30 s median-free moving average excluding center
        k = np.ones(61); k[30]=0; sm = np.convolve(h, k/60, mode="same")
        ex = h - sm
        top = np.argsort(-ex)[:5]
        print(ph, apt, len(y), "top excess values:", [(int(t), int(h[t]), round(sm[t],1)) for t in top if ex[t] > 5*np.sqrt(sm[t]+1)])
