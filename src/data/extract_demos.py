"""Extract .dem files from downloaded HLTV .rar/.zip archives via 7-Zip.

HLTV GOTV downloads arrive as one archive per series (a .rar containing one .dem per
map). This unpacks the .dem files into demos/extracted/. By default it extracts only
maps whose filename contains 'inferno' (our scope); use --all-maps to extract every .dem.

Requires 7-Zip (auto-located; installed at D:\\7-Zip on this machine).

Usage:
    python src/data/extract_demos.py                 # extract inferno demos from demos/raw
    python src/data/extract_demos.py --all-maps      # extract every .dem
    python src/data/extract_demos.py --list          # just list archive contents
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "demos" / "raw"
EXTRACTED = ROOT / "demos" / "extracted"

SEVENZIP_CANDIDATES = [
    r"D:\7-Zip\7z.exe",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]
ARCHIVE_EXTS = {".rar", ".zip", ".7z"}


def find_7z() -> str:
    for c in SEVENZIP_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("7z") or shutil.which("7za")
    if found:
        return found
    sys.exit("7-Zip not found. Install with: winget install 7zip.7zip")


def list_archive(sevenzip: str, archive: Path) -> list[str]:
    """Return the file names inside an archive (technical listing)."""
    out = subprocess.run(
        [sevenzip, "l", "-slt", str(archive)],
        capture_output=True, text=True,
    )
    names = []
    for line in out.stdout.splitlines():
        if line.startswith("Path = "):
            p = line[len("Path = "):].strip()
            if p.lower().endswith(".dem"):
                names.append(p)
    return names


def extract_members(sevenzip: str, archive: Path, members: list[str], dest: Path):
    """Extract specific members flat into dest (-o, e = flat extract)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sevenzip, "e", str(archive), f"-o{dest}", "-y", *members]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [warn] 7z exit {r.returncode}: {r.stderr.strip()[:200]}", file=sys.stderr)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXTRACTED)
    ap.add_argument("--all-maps", action="store_true", help="extract every .dem, not just inferno")
    ap.add_argument("--list", action="store_true", help="only list archive contents")
    args = ap.parse_args()

    sevenzip = find_7z()
    archives = sorted(p for p in args.raw_dir.glob("*") if p.suffix.lower() in ARCHIVE_EXTS)
    if not archives:
        print(f"No archives in {args.raw_dir}")
        return
    print(f"7-Zip: {sevenzip}\nFound {len(archives)} archive(s) in {args.raw_dir}\n")

    total_dem = 0
    for arc in archives:
        dems = list_archive(sevenzip, arc)
        inferno = [d for d in dems if "inferno" in Path(d).name.lower()]
        print(f"[{arc.name}] {len(dems)} .dem inside; {len(inferno)} inferno")
        for d in dems:
            print(f"    - {Path(d).name}")
        if args.list:
            continue
        want = dems if args.all_maps else (inferno or [])
        if not want:
            print("    (no inferno map by filename; use --all-maps to force, "
                  "or the .dem map names may differ)")
            continue
        # skip ones already extracted
        want = [d for d in want if not (args.out / Path(d).name).exists()]
        if want and extract_members(sevenzip, arc, want, args.out):
            total_dem += len(want)
            print(f"    extracted {len(want)} -> {args.out}")

    print(f"\nDone. Extracted {total_dem} new .dem file(s) to {args.out}")


if __name__ == "__main__":
    main()
