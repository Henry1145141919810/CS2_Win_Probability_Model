"""PILOT assembler for the defuse-progress experiment. NOT the paper pipeline.

Why this exists: assemble.py needs ~/.awpy/navs/de_inferno.json for Voronoi map control and
the bomb nav-path distances, and awpy's data mirror (awpycs.com) is dead -- every resource
404s and 2.0.2 is the latest release, so `awpy get navs` cannot be recovered without a copy
of that file. Everything ELSE is nav-free, so this builds the largest table we can without
it: economy + tactical + firepower + defuse-progress.

MISSING vs the real table: Voronoi/grey/territory map control, bomb site+distance geometry,
the defuse-RACE features, and the control interactions. So numbers from this file are NOT
comparable to any published result, and it writes to its own output path. It exists to
answer one question -- does the win-probability curve move correctly while a defuse is
running -- on a small sample, before spending a full re-parse of the training set.

Usage:
    python src/features/assemble_defuse_pilot.py \
        --parquet-root data/holdout2026/parquet_defuse \
        --out data/holdout2026/pilot_defuse.parquet
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
from features.economy import economy_features, ECONOMY_COLS  # noqa: E402
from features.positional import tactical_features, TACTICAL_COLS  # noqa: E402
from features.firepower import firepower_features, FIREPOWER_COLS  # noqa: E402
from features.defuse import (DefuseTracker, defuse_progress_features,  # noqa: E402
                             attempts_by_round, BOMB_PROGRESS_COLS)

PILOT_SETS = {
    "A":    ECONOMY_COLS,                                         # the literature baseline
    "AD":   ECONOMY_COLS + BOMB_PROGRESS_COLS,                    # + defuse progress
    "AT":   ECONOMY_COLS + TACTICAL_COLS,
    "ATF":  ECONOMY_COLS + TACTICAL_COLS + FIREPOWER_COLS,
    "ATFD": ECONOMY_COLS + TACTICAL_COLS + FIREPOWER_COLS + BOMB_PROGRESS_COLS,
}


def assemble_demo(match_id: str, root: Path) -> pl.DataFrame | None:
    rounds = pl.read_parquet(root / "rounds" / f"{match_id}.parquet")
    ticks = pl.read_parquet(root / "ticks" / f"{match_id}.parquet")
    clean = clean_rounds(rounds, ticks)
    if clean is None:
        return None
    bomb_raw = pl.read_parquet(root / "bomb" / f"{match_id}.parquet")
    plants = {r["round_num"]: {"tick": r["tick"]}
              for r in bomb_raw.filter(pl.col("event") == "plant").iter_rows(named=True)}
    df = root / "defuse" / f"{match_id}.parquet"
    defuse_by_round = attempts_by_round(pl.read_parquet(df) if df.exists() else None)

    rows = []
    ct_score = t_score = 0
    for rr in clean.iter_rows(named=True):
        rn = rr["round_num"]
        label = 1 if rr["winner"] == "ct" else 0
        rt = ticks.filter((pl.col("round_num") == rn) & (pl.col("tick") >= rr["freeze_end"])
                          & (pl.col("tick") <= rr["end"]))
        tracker = DefuseTracker(defuse_by_round.get(rn))
        for tick in sorted(rt["tick"].unique().to_list()):
            snap = rt.filter(pl.col("tick") == tick)
            if snap.height < 2:
                continue
            rows.append({
                "match_id": match_id, "tick": tick, "round_num_raw": rn,
                **economy_features(snap, rr, tick, ct_score, t_score),
                **tactical_features(snap),
                **firepower_features(snap, match_id),
                **defuse_progress_features(tracker, plants.get(rn), tick),
                "ct_won": label,
            })
        if rr["winner"] == "ct":
            ct_score += 1
        else:
            t_score += 1
    return pl.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root", type=Path,
                    default=ROOT / "data" / "holdout2026" / "parquet_defuse")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "holdout2026" / "pilot_defuse.parquet")
    args = ap.parse_args()

    demos = [Path(f).stem for f in sorted(glob.glob(str(args.parquet_root / "rounds" / "*.parquet")))]
    parts, skipped = [], []
    for i, m in enumerate(demos, 1):
        d = assemble_demo(m, args.parquet_root)
        if d is None:
            skipped.append(m)
            print(f"[{i}/{len(demos)}] {m}: SKIP (failed validation)")
        else:
            parts.append(d)
            fired = int(d["defuse_in_progress"].sum())
            print(f"[{i}/{len(demos)}] {m}: {d.height} snapshots ({fired} defusing)")
    if not parts:
        print("nothing assembled")
        return
    full = pl.concat(parts, how="vertical")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(args.out)
    print(f"\nWROTE {full.height} x {full.width} -> {args.out}")
    print(f"demos {len(parts)} (skipped {len(skipped)}) | ct_won {full['ct_won'].mean():.3f}")
    print(f"defuse_in_progress rows: {int(full['defuse_in_progress'].sum())} "
          f"({full['defuse_in_progress'].mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
