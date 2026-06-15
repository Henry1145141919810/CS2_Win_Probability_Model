"""Validate parsed-demo round labels, accounting for the halftime side-swap.

Key fact: per-round `winner` is a SIDE ('ct'/'t'), but first-to-13 is per TEAM, and
teams swap sides at halftime. So you cannot judge map completeness from side-win counts
(a clean 13-9 map is ~11-11 by side). This validator maps each round's winning side to
the TEAM that held that side (via `team_clan_name` in the ticks), reconstructs the real
team score, finds the clinch (first to 13 in regulation; OT in +3 blocks), and trims any
phantom rounds recorded after the map was decided.

For modeling, the per-round SIDE winner ('ct'/'t') over the REAL rounds is the label.

Usage:
    python src/data/validate_parquet.py            # report
    python src/data/validate_parquet.py --write     # also write the validation CSV
"""
from __future__ import annotations
import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
OUT_CSV = ROOT / "configs" / "parsed_demos_validation.csv"

WIN = 13
REG_MAX_LOSER = 11


def ct_clan_by_round(ticks: pl.DataFrame) -> dict[int, str]:
    """Map round_num -> the clan playing CT that round (mode of ct-side clans)."""
    out = {}
    df = ticks.filter(pl.col("side") == "ct").drop_nulls("team_clan_name")
    for rn, grp in df.group_by("round_num"):
        rn = rn[0] if isinstance(rn, tuple) else rn
        m = grp["team_clan_name"].mode()
        if len(m):
            out[rn] = m[0]
    return out


def reconstruct(rounds: pl.DataFrame, ticks: pl.DataFrame) -> dict:
    clans = [c for c in ticks["team_clan_name"].drop_nulls().unique().to_list()]
    rounds = rounds.sort("round_num")
    ctclan = ct_clan_by_round(ticks)

    team = Counter()
    side = Counter()
    real_rounds = 0
    clinch = None
    for rr in rounds.iter_rows(named=True):
        rn, wside = rr["round_num"], rr["winner"]
        cc = ctclan.get(rn)
        if cc is None or wside not in ("ct", "t"):
            continue
        others = [c for c in clans if c != cc]
        tclan = others[0] if others else "?"
        win_clan = cc if wside == "ct" else tclan
        team[win_clan] += 1
        side[wside] += 1
        real_rounds += 1
        hi, lo = max(team.values()), (min(team.values()) if len(team) > 1 else 0)
        if hi == WIN and lo <= REG_MAX_LOSER:
            clinch = ("regulation", rn, real_rounds)
            break
        if hi >= 16 and (hi - lo) >= 2 and (hi - WIN) % 3 == 0:
            clinch = ("overtime", rn, real_rounds)
            break

    valid = clinch is not None
    winner_team = max(team, key=team.get) if team else None
    return {
        "valid": valid,
        "mode": clinch[0] if valid else "no-clinch",
        "team_score": "-".join(str(team[c]) for c in clans) if valid else "--",
        "clans": "/".join(clans),
        "winner_team": winner_team if valid else None,
        "real_rounds": clinch[2] if valid else real_rounds,
        "n_phantom": (len(rounds) - clinch[2]) if valid else 0,
        "ct_side_wins": side["ct"],
        "t_side_wins": side["t"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROUNDS_DIR / "*.parquet")))
    if not files:
        print(f"No rounds parquet in {ROUNDS_DIR}")
        return

    rows = []
    print(f"{'demo':42s} {'raw':>3} {'team':>7} {'phan':>4} {'ctW':>3} {'tW':>3} {'mode':>10} ok")
    for f in files:
        stem = os.path.basename(f).replace(".parquet", "")
        tf = TICKS_DIR / f"{stem}.parquet"
        if not tf.exists():
            print(f"{stem[:42]:42s}  (no ticks parquet)")
            continue
        rounds = pl.read_parquet(f)
        ticks = pl.read_parquet(tf, columns=["round_num", "side", "team_clan_name", "tick"])
        res = reconstruct(rounds, ticks)
        print(f"{stem[:42]:42s} {len(rounds):>3} {res['team_score']:>7} {res['n_phantom']:>4} "
              f"{res['ct_side_wins']:>3} {res['t_side_wins']:>3} {res['mode']:>10} "
              f"{'OK' if res['valid'] else 'DROP'}")
        rows.append({"demo": stem, "raw_rounds": len(rounds), **res})

    n_ok = sum(r["valid"] for r in rows)
    n_phantom = sum(1 for r in rows if r["valid"] and r["n_phantom"] > 0)
    print(f"\nUsable: {n_ok}/{len(rows)}  |  phantom-trimmed: {n_phantom}  |  dropped: {len(rows)-n_ok}")
    if args.write and rows:
        pl.DataFrame(rows).write_csv(OUT_CSV)
        print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
