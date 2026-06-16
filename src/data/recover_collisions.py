"""Recover demos lost to extraction filename-collisions (cross-event matchups).

Background: the old extractor named demos by matchup+map (e.g. faze-vs-vitality-m1-
inferno.dem), which is NOT unique across events. When the same matchup played Inferno in
two events, the files collided and one game was silently dropped. This script:

  1. reads .cache/extract.log to map each inferno .dem name -> source archives,
  2. finds CROSS-EVENT collisions (archives with >1 distinct series_id),
  3. for each, deletes the old ambiguous demo + its parquet, then re-extracts every
     distinct series under the collision-safe '<series_id>__<demname>' name.

Afterwards run batch_parse (parses the new files) -> validate -> assemble.
"""
from __future__ import annotations
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
from extract_demos import find_7z, list_archive, extract_members, series_id, RAW, EXTRACTED  # noqa: E402

PARQUET = ROOT / "data" / "parquet"
CHANNELS = ["ticks", "kills", "rounds", "bomb", "grenades"]
LOG = ROOT / ".cache" / "extract.log"


def dem_to_archives() -> dict[str, list[str]]:
    out = defaultdict(list)
    cur = None
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\[(.+\.rar)\]", line)
        if m:
            cur = m.group(1)
        d = re.match(r"\s+- (.*inferno\.dem)", line)
        if d and cur:
            out[d.group(1)].append(cur)
    return out


def main():
    sz = find_7z()
    groups = dem_to_archives()
    recovered = 0
    for dem, archives in groups.items():
        sids = {series_id(a) for a in archives}
        if len(sids) < 2:
            continue  # not cross-event (redundant same-series dups are fine)
        print(f"[collision] {dem}  across {len(sids)} events")
        # delete the old ambiguous plain-named demo + its parquet
        (EXTRACTED / dem).unlink(missing_ok=True)
        stem = dem[:-4]
        for ch in CHANNELS:
            (PARQUET / ch / f"{stem}.parquet").unlink(missing_ok=True)
        # re-extract each distinct series under a unique name
        for arc in sorted(set(archives)):
            members = [m for m in list_archive(sz, RAW / arc) if "inferno" in m.lower()]
            if members:
                extract_members(sz, arc if isinstance(arc, Path) else RAW / arc, members, EXTRACTED)
                recovered += 1
    print(f"\nRe-extracted {recovered} series-id-named demos from cross-event collisions.")
    print("Next: batch_parse -> validate_parquet -> assemble.")


if __name__ == "__main__":
    main()
