"""Build the raw-data supplement for Prof. Wyner.

A full-resolution copy of everything behind the round-level export: the per-second modelling
table, every parsed demo channel for the 220 training matches, and all reference tables.

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
TREE = ROOT / "data" / "parquet_defuse"
OUT = ROOT / "exports" / "cs2_inferno_raw_supplement_v1"
CHANNELS = ["ticks", "rounds", "kills", "bomb", "smokes", "infernos", "defuse"]


def main() -> None:
    tr = pl.read_parquet(ROOT / "data" / "training_dataset.parquet", columns=["match_id"])
    matches = sorted(set(tr["match_id"].unique().to_list()))
    print(f"training matches: {len(matches)}")

    (OUT / "modelling").mkdir(parents=True, exist_ok=True)
    (OUT / "reference").mkdir(parents=True, exist_ok=True)

    # ---- per-second modelling table -------------------------------------------------------
    src = ROOT / "data" / "training_dataset.parquet"
    dst = OUT / "modelling" / "training_dataset_per_second.parquet"
    shutil.copy(src, dst)
    print(f"modelling table -> {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")

    # ---- parsed channels, restricted to the training matches -------------------------------
    manifest = {}
    for ch in CHANNELS:
        d = OUT / "parsed" / ch
        d.mkdir(parents=True, exist_ok=True)
        n, sz, cols = 0, 0, None
        for f in sorted(glob.glob(str(TREE / ch / "*.parquet"))):
            mid = os.path.basename(f)[:-8]
            if mid not in matches:
                continue
            shutil.copy(f, d / os.path.basename(f))
            n += 1
            sz += os.path.getsize(f)
            if cols is None:
                cols = pl.read_parquet(f).columns
        manifest[ch] = {"files": n, "mb": round(sz / 1e6, 1), "columns": cols}
        print(f"  {ch:9s} {n:4d} files  {sz/1e6:7.1f} MB")

    # ---- reference tables -------------------------------------------------------------------
    for f in sorted(glob.glob(str(ROOT / "configs" / "*"))):
        if os.path.isfile(f):
            shutil.copy(f, OUT / "reference" / os.path.basename(f))
    print(f"reference files: {len(list((OUT/'reference').iterdir()))}")

    # ---- machine-readable manifest ----------------------------------------------------------
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "dataset": "CS2 Inferno Raw Supplement v1",
        "scope": "220 de_inferno training matches (2024-2025)",
        "n_matches": len(matches),
        "modelling_table": {
            "file": "modelling/training_dataset_per_second.parquet",
            "rows": pl.read_parquet(src, columns=["match_id"]).height,
        },
        "parsed_channels": manifest,
        "matches": matches,
    }, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nwrote -> {OUT}  ({total/1e6:.0f} MB total)")


if __name__ == "__main__":
    main()
