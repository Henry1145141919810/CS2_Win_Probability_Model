"""Pathwise extreme-path calibration (the FULL Benchmark's trajectory-honesty test).

Treats each round's per-second win-probability as a Doob martingale (p0 = round-start WP, terminal =
outcome) and asks whether its EXTREMES behave like an honest sequential forecast. Based on Pipping &
Wyner, "Conditional Extreme-Path Benchmarks for Sequential Probability Forecasts" (2026) and the
companion note "A Paradox of Blown Leads" (2025).

Objects (per round):
  wp_win_t   = eventual winner's win-prob at second t     (= p_t if CT won else 1 - p_t)
  trough     = min_t wp_win_t                              (eventual winner's lowest point)
  loser_peak = max_t (1 - wp_win_t) = 1 - trough           (eventual loser's highest point)
  p0_loser   = loser's win-prob at freeze-end              (= 1 - wp_win_0)

Benchmark (continuous-path calibrated martingale; conservative in discrete/jumpy CS2):
  P(loser_peak >= x | this side loses) = (p0L/(1-p0L)) * ((1-x)/x),  x in [p0L, 1)
  Winner's trough <= y  <=>  loser_peak >= 1-y.

TASK A: observed fraction (trough <= y) vs the averaged benchmark, thresholds y in {0.05,0.10,0.20},
        with match-level block-bootstrap CIs. Plus sanity checks.
TASK B: PIT U_i = F(loser_peak_i; p0L_i) ~ Unif(0,1) under the null; one-sided KS D_upper (over-
        reaction / upper-tail inflation) and D_lower, with a null-simulated reference.

Usage: python src/models/pathwise_calibration.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
OUTCSV = ROOT / "outputs" / "pathwise_benchmark.csv"
MODELS = [("logreg", "EFB2"), ("lgbm", "EB2")]   # headline (reproduce 7.2%) + production
THRESH = [0.05, 0.10, 0.20]
B = 500


def per_round(df: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Collapse per-snapshot predictions to per-round trajectory extremes."""
    d = (df.select(["match_id", "round_num", "tick", "ct_won"])
           .with_columns(p=pl.Series(p))
           .with_columns(wp_win=pl.when(pl.col("ct_won") == 1).then(pl.col("p")).otherwise(1 - pl.col("p")))
           .sort(["match_id", "round_num", "tick"]))
    g = d.group_by(["match_id", "round_num"], maintain_order=True).agg(
        trough=pl.col("wp_win").min(),
        wp_win0=pl.col("wp_win").first(),
        n=pl.len(),
    )
    g = g.with_columns(
        loser_peak=1 - pl.col("trough"),
        p0_loser=1 - pl.col("wp_win0"),
    )
    return g


def bench_tail_trough(p0L: np.ndarray, y: float) -> np.ndarray:
    """Per-round benchmark P(trough <= y) = P(loser_peak >= 1-y | lose). x = 1-y."""
    x = 1 - y
    out = (p0L / (1 - p0L)) * ((1 - x) / x)      # = (p0L/(1-p0L)) * (y/(1-y))
    out = np.where(x >= p0L, out, 1.0)           # loser already started above threshold -> guaranteed
    return np.clip(out, 0.0, 1.0)


def pit(loser_peak: np.ndarray, p0L: np.ndarray) -> np.ndarray:
    """U = F(loser_peak; p0L), the benchmark CDF conditional on loss."""
    tail = (p0L / (1 - p0L)) * ((1 - loser_peak) / loser_peak)   # P(M >= m | lose)
    U = 1.0 - tail
    U = np.where(loser_peak < p0L, 0.0, U)       # below start -> CDF 0
    return np.clip(U, 0.0, 1.0)


def match_boot(vals_by_match: dict, stat_fn, B=B, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(vals_by_match)
    arrs = [vals_by_match[k] for k in keys]
    out = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        s = np.concatenate([arrs[j] for j in idx])
        out.append(stat_fn(s))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ks_upper_lower(U: np.ndarray):
    Us = np.sort(U)
    n = len(Us)
    ecdf = np.arange(1, n + 1) / n
    t = Us
    d_upper = float(np.max(t - (ecdf - 1.0 / n)))   # sup_t (t - Fhat(t))
    d_lower = float(np.max(ecdf - t))               # sup_t (Fhat(t) - t)
    return d_upper, d_lower


def ks_null_pvalue(n: int, d_upper: float, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(reps):
        u = np.sort(rng.random(n))
        ecdf = np.arange(1, n + 1) / n
        du = np.max(u - (ecdf - 1.0 / n))
        if du >= d_upper:
            cnt += 1
    return (cnt + 1) / (reps + 1)


def main():
    df = pl.read_parquet(TRAIN)
    y = df["ct_won"].to_numpy().astype(float)
    print(f"{df.height} snaps / {df['match_id'].n_unique()} matches / "
          f"{df.select(['match_id','round_num']).unique().height} rounds\n")

    rows = []
    for model, fs in MODELS:
        cols = FEATURE_SETS[fs]
        oof, _ = oof_predict(df, cols, model)
        g = per_round(df, oof)
        trough = g["trough"].to_numpy()
        p0L = g["p0_loser"].to_numpy()
        lpk = g["loser_peak"].to_numpy()
        mid = g["match_id"].to_numpy()
        by = {}
        for i, m in enumerate(mid):
            by.setdefault(m, []).append(i)
        by = {k: np.array(v) for k, v in by.items()}

        print(f"===== {model} {fs} =====")
        # ---- sanity checks ----
        print(f"  loser_peak: mean {lpk.mean():.3f} median {np.median(lpk):.3f} "
              f"(paper NFL/NBA ~0.69)  |  PIT mean {pit(lpk,p0L).mean():.3f} (want ~0.5)")

        # ---- TASK A: observed vs benchmark per threshold ----
        for yv in THRESH:
            obs = float((trough <= yv).mean())
            ben = float(bench_tail_trough(p0L, yv).mean())
            lo, hi = match_boot(
                {k: (trough[v] <= yv).astype(float) for k, v in by.items()},
                lambda s: s.mean())
            print(f"  trough <= {yv:.2f} : observed {obs*100:5.2f}% ({lo*100:.2f},{hi*100:.2f}) "
                  f"| benchmark {ben*100:5.2f}%  -> obs/bench {obs/ben:.2f}")
            rows.append(dict(model=model, fs=fs, threshold=yv, observed=obs,
                             obs_lo=lo, obs_hi=hi, benchmark=ben, ratio=obs / ben))

        # ---- TASK B: PIT + KS ----
        U = pit(lpk, p0L)
        du, dl = ks_upper_lower(U)
        pval = ks_null_pvalue(len(U), du)
        verdict = ("OVER-REACTS (upper-tail inflated)" if pval < 0.05
                   else "no upper-tail inflation (honest / under-powered)")
        print(f"  PIT/KS: D_upper {du:.4f} (p={pval:.4f}) D_lower {dl:.4f}  -> {verdict}\n")
        rows.append(dict(model=model, fs=fs, threshold=-1, observed=du, obs_lo=dl,
                         obs_hi=pval, benchmark=float(U.mean()), ratio=np.nan))

        # dump per-round records for the figure
        g.with_columns(U=pl.Series(U), model=pl.lit(f"{model}_{fs}")).write_parquet(
            ROOT / "outputs" / f"pathwise_perround_{model}_{fs}.parquet")

    pl.DataFrame(rows).write_csv(OUTCSV)
    print(f"wrote {OUTCSV}")


if __name__ == "__main__":
    main()
