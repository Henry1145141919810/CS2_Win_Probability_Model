"""Package a lean data bundle for a collaborator (upload to Drive/Dropbox).

The partner clones the GitHub repo for code, then drops this bundle's contents into the
repo so they're at parity WITHOUT downloading the 300GB of demos or re-parsing. Includes:
  - data/parquet/{ticks,kills,rounds,bomb,smokes,infernos}  (used channels; no grenades)
  - data/training_dataset.parquet                           (the model-ready table)
  - configs/demo_list_final.csv                             (exact demos in the model)
  - .cache/visibility_de_inferno.npy                        (LOS matrix; saves a ~2h build)

Output: cs2_share_bundle.zip  (~80-100 MB)

Usage: python src/data/make_share_bundle.py
"""
from __future__ import annotations
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cs2_share_bundle.zip"
CHANNELS = ["ticks", "kills", "rounds", "bomb", "smokes", "infernos"]


def main():
    items: list[Path] = []
    for ch in CHANNELS:
        items += sorted((ROOT / "data" / "parquet" / ch).glob("*.parquet"))
    for extra in ["data/training_dataset.parquet", "configs/demo_list_final.csv",
                  ".cache/visibility_de_inferno.npy"]:
        p = ROOT / extra
        if p.exists():
            items.append(p)
    items += sorted((ROOT / "outputs" / "figures").glob("*.png"))  # current figures
    items += sorted((ROOT / "outputs" / "figures").glob("*.gif"))

    total = sum(p.stat().st_size for p in items) / 1e6
    print(f"bundling {len(items)} files (~{total:.0f} MB) -> {OUT.name}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in items:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"done -> {OUT}  ({OUT.stat().st_size/1e6:.0f} MB compressed)")
    print("Partner: unzip into the repo root (keeps data/, configs/, .cache/ paths).")


if __name__ == "__main__":
    main()
