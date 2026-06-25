"""Build a COMPACT per-player trajectory dataset for the GAT (spatial deep model).

The aggregate training_dataset.parquet collapses 10 players into summary stats. A graph/
attention model wants the RAW per-player state. This script reads the parsed ticks and emits,
for each per-second snapshot, up to 10 alive-player "nodes" with their raw movement/state:
    x, y, vx, vy, yaw, hp, armor, equip, kit, is_ct   (10 features/player)
flattened into fixed 10 slots (p0..p9) + an alive mask, plus the round label and a little
context (time, bomb_planted, score_diff). Permutation-invariant model -> slot order doesn't matter.

Much smaller than the full ticks (only the needed columns), so it can be uploaded to Betty.

Usage:
    python src/features/build_trajectory_dataset.py            # all demos
    python src/features/build_trajectory_dataset.py --limit 3  # quick test
Output: data/trajectory_dataset.parquet
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.validate_parquet import clean_rounds  # noqa: E402

ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
OUT = ROOT / "data" / "trajectory_dataset.parquet"
MAXP = 10
PFEATS = ["x", "y", "vx", "vy", "yaw", "hp", "armor", "equip", "kit", "isct"]


def assemble_demo(match_id: str) -> pl.DataFrame | None:
    rounds = pl.read_parquet(ROUNDS_DIR / f"{match_id}.parquet")
    ticks = pl.read_parquet(TICKS_DIR / f"{match_id}.parquet")
    clean = clean_rounds(rounds, ticks)
    if clean is None:
        return None
    rows = []
    ct_score = t_score = 0
    for rr in clean.iter_rows(named=True):
        rn = rr["round_num"]
        label = 1 if rr["winner"] == "ct" else 0
        rt = ticks.filter((pl.col("round_num") == rn) & (pl.col("tick") >= rr["freeze_end"])
                          & (pl.col("tick") <= rr["end"]))
        for tick in sorted(rt["tick"].unique().to_list()):
            snap = rt.filter((pl.col("tick") == tick) & (pl.col("health") > 0))
            if snap.height < 2:
                continue
            row = {"match_id": match_id, "round_num": rn, "tick": tick, "ct_won": label,
                   "time_elapsed_sec": (tick - rr["freeze_end"]) / 64.0,
                   "ct_score": ct_score, "t_score": t_score}
            xs = snap["X"].to_list(); ys = snap["Y"].to_list()
            vx = snap["velocity_X"].to_list(); vy = snap["velocity_Y"].to_list()
            yaw = snap["yaw"].to_list(); hp = snap["health"].to_list()
            arm = snap["armor"].to_list(); eq = snap["current_equip_value"].to_list()
            kit = snap["has_defuser"].to_list() if "has_defuser" in snap.columns else [0] * snap.height
            side = snap["side"].to_list()
            vals = {"x": xs, "y": ys, "vx": vx, "vy": vy, "yaw": yaw, "hp": hp,
                    "arm": arm, "equip": eq, "kit": kit, "side": side}
            def _f(seq, i):                      # None-safe float (some velocities are null)
                v = seq[i]
                return float(v) if v is not None else 0.0
            for i in range(MAXP):
                ok = i < snap.height
                row[f"p{i}_alive"] = 1 if ok else 0
                row[f"p{i}_x"] = _f(vals["x"], i) if ok else 0.0
                row[f"p{i}_y"] = _f(vals["y"], i) if ok else 0.0
                row[f"p{i}_vx"] = _f(vals["vx"], i) if ok else 0.0
                row[f"p{i}_vy"] = _f(vals["vy"], i) if ok else 0.0
                row[f"p{i}_yaw"] = _f(vals["yaw"], i) if ok else 0.0
                row[f"p{i}_hp"] = _f(vals["hp"], i) if ok else 0.0
                row[f"p{i}_armor"] = _f(vals["arm"], i) if ok else 0.0
                row[f"p{i}_equip"] = _f(vals["equip"], i) if ok else 0.0
                row[f"p{i}_kit"] = float(bool(vals["kit"][i])) if ok else 0.0
                row[f"p{i}_isct"] = 1.0 if (ok and str(vals["side"][i]).lower().startswith("c")) else 0.0
            rows.append(row)
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
    excl_file = ROOT / "configs" / "excluded_offlist.txt"
    if excl_file.exists():
        excl = {ln.strip() for ln in excl_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        demos = [d for d in demos if d not in excl]
    if args.limit:
        demos = demos[: args.limit]

    parts = []
    for i, mtch in enumerate(demos, 1):
        df = assemble_demo(mtch)
        if df is not None:
            parts.append(df)
            print(f"[{i}/{len(demos)}] {mtch}: {df.height} snapshots")
        else:
            print(f"[{i}/{len(demos)}] {mtch}: SKIP")
    full = pl.concat(parts, how="vertical")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(args.out)
    sz = args.out.stat().st_size / 1e6
    print(f"\nWROTE {full.height} snapshots x {full.width} cols -> {args.out} ({sz:.0f} MB)")
    print(f"label balance ct_won: {full['ct_won'].mean():.3f}; matches: {full['match_id'].n_unique()}")


if __name__ == "__main__":
    main()
