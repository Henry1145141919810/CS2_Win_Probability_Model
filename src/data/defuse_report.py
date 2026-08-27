"""Summarise the derived `defuse` channel across a parsed tree.

Answers the question that decides whether defuse-progress is a real feature or a
relabelling of `ct_won`: how many defuse ATTEMPTS were interrupted? A tree with only
completed defuses would make "someone started defusing" equivalent to "CT won".

Also reports the duration distribution, which is a free data-quality check: a defuse
with a kit is exactly 5.000 s (320 ticks) and without one 10.000 s, so completed
attempts must cluster on those two values.

Usage:
    python src/data/defuse_report.py --tree data/holdout2026/parquet_defuse
"""
from __future__ import annotations
import argparse
from pathlib import Path

import polars as pl

TICKRATE = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True, help="parsed parquet tree")
    args = ap.parse_args()

    files = sorted((args.tree / "defuse").glob("*.parquet"))
    if not files:
        print(f"no defuse channel in {args.tree} — was it parsed with is_defusing?")
        return
    df = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("match")) for f in files])
    df = df.with_columns(((pl.col("end_tick") - pl.col("start_tick")) / TICKRATE).alias("dur"))

    n, done = df.height, int(df["completed"].sum())
    print(f"tree      : {args.tree}")
    print(f"demos     : {len(files)}")
    print(f"attempts  : {n}")
    print(f"completed : {done}")
    print(f"INTERRUPTED: {n - done}  ({(n - done) / n * 100:.1f}%)   <- the non-leaky rows")

    print(f"\nby kit:")
    for kit in (True, False):
        s = df.filter(pl.col("had_kit") == kit)
        if not s.height:
            continue
        d = int(s["completed"].sum())
        print(f"  {'with kit (5s)' if kit else 'no kit (10s)':<15} {s.height:>4} attempts, "
              f"{d} completed, {s.height - d} interrupted")

    print(f"\nduration of COMPLETED attempts (should sit on 5.0 / 10.0):")
    c = df.filter(pl.col("completed") == 1)
    for kit in (True, False):
        s = c.filter(pl.col("had_kit") == kit)
        if not s.height:
            continue
        print(f"  {'kit' if kit else 'no kit':<8} n={s.height:<4} "
              f"min={s['dur'].min():.3f} median={s['dur'].median():.3f} max={s['dur'].max():.3f}")

    print(f"\nhow far interrupted attempts got before they died:")
    i = df.filter(pl.col("completed") == 0).with_columns(
        (pl.col("dur") / pl.when(pl.col("had_kit")).then(5.0).otherwise(10.0)).alias("frac"))
    if i.height:
        print(f"  n={i.height}  median progress={i['frac'].median():.2f}  max={i['frac'].max():.2f}")
        print(i.select(["match", "round_num", "dur", "had_kit", "frac"]).sort("frac", descending=True).head(10))

    # snapshots the feature touches = total attempt seconds (ticks are sampled at 1 Hz)
    tot = df["dur"].sum()
    interrupted_secs = df.filter(pl.col("completed") == 0)["dur"].sum()
    print(f"\nsnapshots where the feature fires: ~{tot:.0f} over {df['match'].n_unique()} demos"
          f"  ({interrupted_secs:.0f} of them from INTERRUPTED attempts)")


if __name__ == "__main__":
    main()
