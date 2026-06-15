"""Reconcile downloaded .rar archives against the Inferno match list.

Matches each archive in demos/raw to its row in configs/inferno_matches_liquipedia.csv
(by event year + distinctive event keyword + both team slugs in the filename), then sets
the `downloaded` and `rar_filename` columns. Prints a per-event downloaded/total summary
and lists what is still missing.

Usage:
    python src/data/reconcile_downloads.py            # report only
    python src/data/reconcile_downloads.py --write     # also update the CSV
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "demos" / "raw"
CSV = ROOT / "configs" / "inferno_matches_liquipedia.csv"

# Filler tokens dropped from an event name; the rest (organizer, city, year, season#,
# spring/fall/world) must ALL appear in the filename for an event to match.
FILLER = {"intel", "extreme", "masters", "pro", "league", "premier", "major", "cs2",
          "the", "final", "finals", "stage", "playoff", "playoffs", "group", "opening",
          "elimination", "napoca"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Team equivalence groups: Liquipedia aliases (in the CSV) vs HLTV filename slugs.
GROUPS = [
    {"tl", "liquid", "teamliquid"}, {"navi", "natusvincere"}, {"vp", "virtuspro"},
    {"c9", "cloud9"}, {"lvg", "lynnvision"}, {"imp", "imperial"},
    {"themongolz", "mongolz"}, {"vit", "vitality"}, {"ef", "eternalfire"},
    {"rebels", "rebelsgaming"}, {"amkal", "amkalesports"}, {"passionua", "passion"},
    {"g2", "g2esports"}, {"faze", "fazeclan"}, {"spirit", "teamspirit"},
    {"gl", "gamerlegion"},
]


def clean_name(rar: str) -> str:
    """Drop the trailing random hash (everything from the bo#/inferno/m# format marker)
    so short alias tokens (vp, gl, tl) can't false-match inside the hash."""
    base = rar[:-4] if rar.endswith(".rar") else rar
    return re.split(r"-(?:bo\d|inferno|m\d)(?=-|$)", base, maxsplit=1)[0]


def team_in(team: str, nf: str) -> bool:
    """True if `team` (CSV name) is present in normalized filename `nf`, honoring aliases."""
    n = norm(team)
    for g in GROUPS:
        if n in g:
            return any(s in nf for s in g)
    return n in nf


def event_tokens(event: str) -> list[str]:
    """Distinctive event tokens that must ALL be present in a matching filename."""
    return [t for t in re.split(r"[^a-z0-9]", event.lower()) if t and t not in FILLER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df = pl.read_parquet(CSV) if CSV.suffix == ".parquet" else pl.read_csv(CSV)
    rows = df.to_dicts()
    rars = sorted(p.name for p in RAW.glob("*.rar"))
    nf_by_rar = {r: norm(clean_name(r)) for r in rars}

    matched_rar = {}      # rar -> row index
    for i, row in enumerate(rows):
        row["downloaded"] = ""
        row["rar_filename"] = row.get("rar_filename") or ""
    used = set()
    for r, nf in nf_by_rar.items():
        hits = []
        for i, row in enumerate(rows):
            if i in used:
                continue
            if not all(t in nf for t in event_tokens(row["event"])):
                continue
            if team_in(row["team_a"], nf) and team_in(row["team_b"], nf):
                hits.append(i)
        if len(hits) == 1:
            i = hits[0]
            rows[i]["downloaded"] = "Y"
            rows[i]["rar_filename"] = r
            used.add(i)
            matched_rar[r] = i
        # 0 or >1 hits -> leave unmatched (reported below)

    unmatched = [r for r in rars if r not in matched_rar]

    # --- summary ---
    per_event = defaultdict(lambda: [0, 0])
    for row in rows:
        per_event[row["event"]][1] += 1
        if row["downloaded"] == "Y":
            per_event[row["event"]][0] += 1
    n_dl = sum(1 for row in rows if row["downloaded"] == "Y")
    print(f"Matched {len(matched_rar)}/{len(rars)} archives -> {n_dl}/{len(rows)} maps downloaded\n")
    print(f"{'downloaded/total':>16}  event")
    for ev, (d, t) in sorted(per_event.items(), key=lambda x: -x[1][1]):
        bar = "#" * d + "-" * (t - d)
        print(f"{d:>7}/{t:<8} {bar[:30]:30s} {ev}")

    if unmatched:
        print(f"\n{len(unmatched)} archive(s) could not be auto-matched (check manually):")
        for r in unmatched:
            print("   ", r)

    if args.write:
        pl.DataFrame(rows).write_csv(CSV)
        print(f"\nUpdated {CSV}")
    else:
        print("\n(report only; pass --write to update the CSV)")


if __name__ == "__main__":
    main()
