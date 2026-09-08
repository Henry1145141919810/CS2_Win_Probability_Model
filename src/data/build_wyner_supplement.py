"""Build the raw-data supplement for Prof. Wyner.

A full-resolution copy of everything behind the round-level export, split into the two eras the
project keeps strictly apart:

  training_2024_2025/  the 220 de_inferno matches the models are fitted on
  holdout_2026/        the 27-match out-of-time test set (touch-once)

Each era ships its per-second modelling table(s) plus every parsed demo channel. Reference
lookup tables are shared and sit at the top level.

Run:  python src/data/build_wyner_supplement.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exports" / "cs2_inferno_raw_supplement_v1"
CHANNELS = ["ticks", "rounds", "kills", "bomb", "smokes", "infernos", "defuse"]

# modelling tables copied per era, with the destination name spelled out so the file is
# self-describing once it leaves the repo
TRAIN_TABLES = {
    "data/training_dataset.parquet": "training_dataset_per_second.parquet",
    "data/trajectory_dataset.parquet": "trajectory_dataset_per_player.parquet",
}
HOLDOUT_TABLES = {
    "data/test_dataset_2026.parquet": "test_dataset_2026_per_second.parquet",
    "data/test_dataset_2026_defuse.parquet": "test_dataset_2026_with_defuse.parquet",
    "data/test_dataset_2026_lag2025.parquet": "test_dataset_2026_firepower_lagged2025.parquet",
    "data/test_dataset_2026_sameyr.parquet": "test_dataset_2026_firepower_sameyear.parquet",
}


def copy_channels(tree: Path, matches: set[str], dest: Path) -> dict:
    """Copy every parsed channel for `matches` from `tree` into `dest/<channel>/`."""
    out = {}
    for ch in CHANNELS:
        src_dir = tree / ch
        if not src_dir.exists():
            continue
        d = dest / ch
        d.mkdir(parents=True, exist_ok=True)
        n, sz, cols = 0, 0, None
        for f in sorted(glob.glob(str(src_dir / "*.parquet"))):
            if os.path.basename(f)[:-8] not in matches:
                continue
            shutil.copy(f, d / os.path.basename(f))
            n += 1
            sz += os.path.getsize(f)
            if cols is None:
                cols = pl.read_parquet(f).columns
        out[ch] = {"files": n, "mb": round(sz / 1e6, 1), "columns": cols}
        print(f"    {ch:9s} {n:4d} files  {sz/1e6:7.1f} MB")
    return out


def copy_tables(mapping: dict[str, str], dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for src, name in mapping.items():
        p = ROOT / src
        if not p.exists():
            print(f"    [warn] missing {src}")
            continue
        shutil.copy(p, dest / name)
        df = pl.read_parquet(p)
        out[name] = {"rows": df.height, "cols": len(df.columns),
                     "mb": round(p.stat().st_size / 1e6, 1)}
        print(f"    {name:46s} {df.height:>7,} x {len(df.columns):>3}  "
              f"{p.stat().st_size/1e6:6.1f} MB")
    return out


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)          # rebuild cleanly so no stale layout survives
    OUT.mkdir(parents=True)

    manifest = {"dataset": "CS2 Inferno Raw Supplement v1", "eras": {}}

    # ---------------------------------------------------------------- training 2024-2025
    train_matches = set(pl.read_parquet(ROOT / "data" / "training_dataset.parquet",
                                        columns=["match_id"])["match_id"].unique().to_list())
    print(f"training_2024_2025: {len(train_matches)} matches")
    tdir = OUT / "training_2024_2025"
    tables = copy_tables(TRAIN_TABLES, tdir / "modelling")
    chans = copy_channels(ROOT / "data" / "parquet_defuse", train_matches, tdir / "parsed")
    manifest["eras"]["training_2024_2025"] = {
        "n_matches": len(train_matches), "modelling": tables,
        "parsed_channels": chans, "matches": sorted(train_matches)}

    # ---------------------------------------------------------------------- holdout 2026
    hold_matches = set(pl.read_parquet(ROOT / "data" / "test_dataset_2026_defuse.parquet",
                                       columns=["match_id"])["match_id"].unique().to_list())
    print(f"\nholdout_2026: {len(hold_matches)} matches")
    hdir = OUT / "holdout_2026"
    htables = copy_tables(HOLDOUT_TABLES, hdir / "modelling")
    hchans = copy_channels(ROOT / "data" / "holdout2026" / "parquet_defuse",
                           hold_matches, hdir / "parsed")
    manifest["eras"]["holdout_2026"] = {
        "n_matches": len(hold_matches), "modelling": htables,
        "parsed_channels": hchans, "matches": sorted(hold_matches)}

    # ------------------------------------------------------------------ shared reference
    (OUT / "reference").mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(str(ROOT / "configs" / "*"))):
        if os.path.isfile(f):
            shutil.copy(f, OUT / "reference" / os.path.basename(f))
    n_ref = len(list((OUT / "reference").iterdir()))
    manifest["reference_files"] = n_ref
    print(f"\nreference: {n_ref} files")

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nwrote -> {OUT}  ({total/1e6:.0f} MB total)")


if __name__ == "__main__":
    main()
