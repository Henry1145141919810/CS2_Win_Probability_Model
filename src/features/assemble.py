"""Assemble the flat training dataset from parsed + validated demos.

For each demo that passes round validation, walk its REAL (clinch-trimmed) rounds and,
at each per-second snapshot from freeze_end to round end, compute:
  - Pillar 1 (economy/combat)  -> Model A baseline features
  - Pillar 2 (Voronoi map control) -> adds Model B
and attach the label `ct_won` (1 if the round's winning side is CT).

Output: data/training_dataset.parquet  (one row per snapshot)
        + match_id column for GroupKFold (never split within a match).

Usage:
    python src/features/assemble.py            # all parsed demos
    python src/features/assemble.py --limit 3  # quick test
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.validate_parquet import clean_rounds  # noqa: E402
from features.economy import economy_features  # noqa: E402
from features.mapcontrol import control_features, control_trend, control_volatility  # noqa: E402
from features.positional import tactical_features  # noqa: E402
from features.bomb import plant_info, bomb_features  # noqa: E402

ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
BOMB_DIR = ROOT / "data" / "parquet" / "bomb"
OUT = ROOT / "data" / "training_dataset.parquet"


def assemble_demo(match_id: str) -> pl.DataFrame | None:
    rounds = pl.read_parquet(ROUNDS_DIR / f"{match_id}.parquet")
    ticks = pl.read_parquet(TICKS_DIR / f"{match_id}.parquet")
    clean = clean_rounds(rounds, ticks)
    if clean is None:
        return None
    bomb_plants = plant_info(pl.read_parquet(BOMB_DIR / f"{match_id}.parquet"))

    rows = []
    ct_score = t_score = 0  # cumulative side wins BEFORE current round
    for rr in clean.iter_rows(named=True):
        rn = rr["round_num"]
        label = 1 if rr["winner"] == "ct" else 0
        rt = ticks.filter((pl.col("round_num") == rn) & (pl.col("tick") >= rr["freeze_end"])
                          & (pl.col("tick") <= rr["end"]))
        ctrl_series = []  # CT overall control over the round, for trend/volatility
        for tick in sorted(rt["tick"].unique().to_list()):
            snap = rt.filter(pl.col("tick") == tick)
            if snap.height < 2:
                continue
            alive = snap.filter(pl.col("health") > 0)
            feats = economy_features(snap, rr, tick, ct_score, t_score)
            mc = control_features(alive["X"].to_list(), alive["Y"].to_list(),
                                  alive["side"].to_list())
            ctrl_series.append(mc["ct_voronoi_control_pct"])
            mc["control_trend"] = control_trend(ctrl_series)
            mc["control_volatility"] = control_volatility(ctrl_series)
            tac = tactical_features(snap)
            bmb = bomb_features(snap, bomb_plants.get(rn), tick)
            rows.append({"match_id": match_id, "tick": tick, **feats, **mc, **tac, **bmb,
                         "ct_won": label})
        # update running score AFTER the round
        if rr["winner"] == "ct":
            ct_score += 1
        else:
            t_score += 1
    return pl.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    demos = [Path(f).stem for f in sorted(glob.glob(str(ROUNDS_DIR / "*.parquet")))]
    if args.limit:
        demos = demos[: args.limit]

    parts, skipped = [], []
    for i, m in enumerate(demos, 1):
        df = assemble_demo(m)
        if df is None:
            skipped.append(m)
            print(f"[{i}/{len(demos)}] {m}: SKIP (failed validation)")
        else:
            parts.append(df)
            print(f"[{i}/{len(demos)}] {m}: {df.height} snapshots")

    if not parts:
        print("No usable demos.")
        return
    full = pl.concat(parts, how="vertical")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(args.out)
    print(f"\nWROTE {full.height} snapshots x {full.width} cols -> {args.out}")
    print(f"demos used: {len(parts)}, skipped: {len(skipped)}")
    print(f"label balance ct_won: {full['ct_won'].mean():.3f}")
    print(f"matches (GroupKFold groups): {full['match_id'].n_unique()}")


if __name__ == "__main__":
    main()
