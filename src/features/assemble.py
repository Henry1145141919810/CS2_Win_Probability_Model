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
from features.mapcontrol import (control_features, control_trend,  # noqa: E402
                                 control_volatility, contest_control, TerritoryControl)
from features.positional import tactical_features  # noqa: E402
from features.bomb import plant_info, bomb_features  # noqa: E402

ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
BOMB_DIR = ROOT / "data" / "parquet" / "bomb"
SMOKES_DIR = ROOT / "data" / "parquet" / "smokes"
OUT = ROOT / "data" / "training_dataset.parquet"
SMOKE_DUR_TICKS = 18 * 64  # CS2 smoke ~18s of vision block
INTERACTION_COLS = ["ctrl_x_eveneco", "terr_x_eveneco",
                    "ctrl_x_equalalive", "terr_x_equalalive"]


def _smokes_by_round(match_id):
    """round_num -> list of (start_tick, end_tick, x, y) for vision-blocking smokes."""
    f = SMOKES_DIR / f"{match_id}.parquet"
    out = {}
    if not f.exists():
        return out
    for r in pl.read_parquet(f).iter_rows(named=True):
        s = r["start_tick"]
        e = r["end_tick"] if r["end_tick"] is not None else s + SMOKE_DUR_TICKS
        out.setdefault(r["round_num"], []).append((s, e, r["X"], r["Y"]))
    return out


def assemble_demo(match_id: str) -> pl.DataFrame | None:
    rounds = pl.read_parquet(ROUNDS_DIR / f"{match_id}.parquet")
    ticks = pl.read_parquet(TICKS_DIR / f"{match_id}.parquet")
    clean = clean_rounds(rounds, ticks)
    if clean is None:
        return None
    bomb_plants = plant_info(pl.read_parquet(BOMB_DIR / f"{match_id}.parquet"))
    smokes_by_round = _smokes_by_round(match_id)

    rows = []
    ct_score = t_score = 0  # cumulative side wins BEFORE current round
    for rr in clean.iter_rows(named=True):
        rn = rr["round_num"]
        label = 1 if rr["winner"] == "ct" else 0
        rt = ticks.filter((pl.col("round_num") == rn) & (pl.col("tick") >= rr["freeze_end"])
                          & (pl.col("tick") <= rr["end"]))
        ctrl_series = []  # CT overall control over the round, for trend/volatility
        terr = TerritoryControl()  # stateful map control w/ memory+decay (per round)
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
            active_smokes = [(x, y) for (s, e, x, y) in smokes_by_round.get(rn, [])
                             if s <= tick < e]
            yaws = alive["yaw"].to_list() if "yaw" in alive.columns else None
            ter = terr.update(  # territory WITH MEMORY + DECAY (held space persists)
                alive["X"].to_list(), alive["Y"].to_list(), alive["side"].to_list(),
                yaws=yaws, smokes=active_smokes, tick=tick)
            tac = tactical_features(snap)
            bmb = bomb_features(snap, bomb_plants.get(rn), tick)
            # interactions: control matters MORE when the round is even (else economy decides)
            even_eco = 1.0 - min(1.0, abs(feats["ct_equipment_value"]
                                          - feats["t_equipment_value"]) / 4000.0)
            equal_alive = float(feats["ct_players_alive"] == feats["t_players_alive"])
            cd = mc.get("control_deficit", mc["ct_voronoi_control_pct"] - 0.5)
            inter = {
                "ctrl_x_eveneco": cd * even_eco,
                "terr_x_eveneco": ter["ct_terr_deficit"] * even_eco,
                "ctrl_x_equalalive": cd * equal_alive,
                "terr_x_equalalive": ter["ct_terr_deficit"] * equal_alive,
            }
            rows.append({"match_id": match_id, "tick": tick, **feats, **mc, **ter,
                         **inter, **tac, **bmb, "ct_won": label})
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
    # drop manually-excluded demos (e.g. qualifier / non-standard games)
    exclude_file = ROOT / "configs" / "excluded_offlist.txt"
    if exclude_file.exists():
        excl = {ln.strip() for ln in exclude_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(demos)
        demos = [d for d in demos if d not in excl]
        print(f"excluded {before - len(demos)} demos via {exclude_file.name}")
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
