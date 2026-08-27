"""Stream demos straight out of an HLTV archive bundle: zip -> rar -> .dem -> parquet.

The macOS/Linux counterpart to extract_demos.py, which shells out to Windows 7-Zip. This
one uses the system `bsdtar` (libarchive >= 3.4 reads RAR5), so nothing has to be installed
-- verified on macOS with bsdtar 3.5.3.

Why streaming: the 2026 bundle is ~23 GB of .rar and the .dem inside are 2-3x that, so
unpacking everything at once would need ~70 GB. This handles ONE archive at a time -- pull
it out of the zip, extract only the maps we want, parse, delete -- so peak extra disk is
about one series (~1.5 GB) no matter how big the bundle is.

Demos are named '<series_id>__<demname>' (extract_demos.series_id), the same collision-safe
scheme the existing parsed tree uses: the same matchup on the same map in two events would
otherwise silently overwrite one another.

Usage:
    python src/data/parse_from_zip.py --zip ~/Downloads/raw.zip \
        --out data/holdout2026/parquet_defuse
    python src/data/parse_from_zip.py --zip ... --out ... --limit 2   # try two first
"""
from __future__ import annotations
import argparse
import gc
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_parse  # noqa: E402
from extract_demos import series_id  # noqa: E402

ARCHIVE_EXTS = {".rar", ".zip", ".7z"}


def _tar(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["tar", *args], capture_output=True, text=True)


def rar_members(archive: Path) -> list[str]:
    """The .dem entries inside an archive, via bsdtar."""
    r = _tar("-tf", str(archive))
    if r.returncode != 0:
        raise RuntimeError(f"cannot read {archive.name}: {r.stderr.strip()[:200]}")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip().lower().endswith(".dem")]


def extract_member(archive: Path, member: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    r = _tar("-xf", str(archive), "-C", str(dest), member)
    if r.returncode != 0:
        raise RuntimeError(f"extract failed {member}: {r.stderr.strip()[:200]}")
    return dest / Path(member).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="bundle of per-series archives")
    ap.add_argument("--out", type=Path, required=True, help="parquet tree to write")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch dir for one archive at a time (default: <out>/_work)")
    ap.add_argument("--map-filter", default="inferno",
                    help="only extract demos whose filename contains this (default: inferno)")
    ap.add_argument("--all-maps", action="store_true")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    work = args.work or (args.out / "_work")
    work.mkdir(parents=True, exist_ok=True)
    keep = None if args.all_maps else args.map_filter.lower()

    with zipfile.ZipFile(args.zip) as z:
        entries = sorted(n for n in z.namelist() if Path(n).suffix.lower() in ARCHIVE_EXTS)
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} archive(s) in {args.zip.name}\n", flush=True)

    counts = {"ok": 0, "skip": 0, "fail": 0, "nomap": 0}
    for i, entry in enumerate(entries, 1):
        name = Path(entry).name
        sid = series_id(name)
        print(f"[{i}/{len(entries)}] {name}", flush=True)
        rar = work / name
        try:
            t0 = time.time()
            with zipfile.ZipFile(args.zip) as z, z.open(entry) as src, open(rar, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            members = rar_members(rar)
            want = [m for m in members if keep is None or keep in Path(m).name.lower()]
            if not want:
                print(f"    no '{keep}' map here ({len(members)} demos) - skipped", flush=True)
                counts["nomap"] += 1
                continue
            for m in want:
                stem = f"{sid}__{Path(m).stem}"
                if (args.out / "ticks" / f"{stem}.parquet").exists() and not args.overwrite:
                    print(f"    {stem}: already parsed - skip", flush=True)
                    counts["skip"] += 1
                    continue
                dem = extract_member(rar, m, work)
                target = work / f"{stem}.dem"
                dem.replace(target)          # collision-safe name -> parse_one uses .stem
                batch_parse._wait_for_memory()
                status = batch_parse.parse_one(target, args.out, args.stride, args.overwrite)
                print(f"    {stem}: {status} ({time.time() - t0:.0f}s)", flush=True)
                counts[status] = counts.get(status, 0) + 1
                target.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            counts["fail"] += 1
            print(f"    [FAIL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            rar.unlink(missing_ok=True)
            for leftover in work.glob("*.dem"):
                leftover.unlink(missing_ok=True)
            gc.collect()

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDone: {counts}")


if __name__ == "__main__":
    main()
