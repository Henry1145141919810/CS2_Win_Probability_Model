"""Craft the final demo list keyed to the EXACT demos fed into the model.

Source of truth = the match_ids in data/training_dataset.parquet (the validated, assembled
demos). For each, attach authoritative game facts from parsed_demos_validation.csv (teams,
team score, winner, real rounds) and, where mappable, event/date/HLTV link from the curated
list. Demos not in the curated 279-list are kept and marked '(off-list)'.

Output: configs/demo_list_final.csv  (one row per model demo).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
from extract_demos import series_id  # noqa: E402

TD = ROOT / "data" / "training_dataset.parquet"
VAL = ROOT / "configs" / "parsed_demos_validation.csv"
CURATED = ROOT / "configs" / "inferno_matches_liquipedia.csv"
LOG = ROOT / ".cache" / "extract.log"
OUT = ROOT / "configs" / "demo_list_final.csv"


def _dem_to_rars() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\[(.+\.rar)\]", line)
        if m:
            cur = m.group(1)
        d = re.match(r"\s+- (.*inferno\.dem)", line)
        if d and cur:
            out.setdefault(d.group(1), []).append(cur)
    return out


def main():
    model_ids = set(pl.read_parquet(TD, columns=["match_id"])["match_id"].unique().to_list())
    val = {r["demo"]: r for r in pl.read_csv(VAL).to_dicts()}
    curated = pl.read_csv(CURATED).to_dicts()
    dem2rars = _dem_to_rars()
    # map source archive -> curated row (reconcile stored rar_filename per matched row)
    by_rar = {r["rar_filename"]: r for r in curated if r.get("rar_filename")}
    by_sid = {}
    for r in curated:
        if r.get("rar_filename"):
            by_sid.setdefault(series_id(r["rar_filename"]), r)

    def attribute(demo):
        """Return the curated row for this demo via its true source archive."""
        if "__" in demo:                       # collision-recovered: event in the id prefix
            return by_sid.get(demo.split("__")[0])
        for rar in dem2rars.get(demo + ".dem", []):   # plain: look up the source archive
            if rar in by_rar:
                return by_rar[rar]
            if series_id(rar) in by_sid:
                return by_sid[series_id(rar)]
        return None

    rows, off = [], 0
    for demo in sorted(model_ids):
        v = val.get(demo, {})
        clans = (v.get("clans") or "").split("/")
        best = attribute(demo)
        if best:
            rows.append({
                "demo_id": demo, "event": best["event"], "stage": best["stage"],
                "series_date": best["series_date"],
                "team_a": best["team_a"], "team_b": best["team_b"],
                "team_score": v.get("team_score", ""), "winner_team": v.get("winner_team", ""),
                "real_rounds": v.get("real_rounds", ""),
                "hltv_match_url": best["hltv_match_url"],
            })
        else:
            off += 1
            ta, tb = (clans + ["?", "?"])[:2]
            rows.append({
                "demo_id": demo, "event": "(off-list)", "stage": "", "series_date": "",
                "team_a": ta, "team_b": tb,
                "team_score": v.get("team_score", ""), "winner_team": v.get("winner_team", ""),
                "real_rounds": v.get("real_rounds", ""), "hltv_match_url": "",
            })

    out = pl.DataFrame(rows).sort(["series_date", "event", "demo_id"])
    out.write_csv(OUT)
    print(f"WROTE {out.height} model demos -> {OUT}  ({off} off-list)")
    from collections import Counter
    for e, n in Counter(r["event"] for r in rows).most_common():
        print(f"  {n:3d}  {e}")


if __name__ == "__main__":
    main()
