"""Craft the final demo list keyed to the EXACT demos fed into the model.

Source of truth = match_ids in data/training_dataset.parquet. For each demo:
  - authoritative teams/score/winner/rounds from parsed_demos_validation.csv,
  - event/date/HLTV from the demo's true source archive (extract log -> rar -> curated row),
  - if the exact match isn't a curated playoff row, attribute the EVENT from the source
    archive (Tier-1 group-stage games) and mark the stage,
  - only genuinely unknown-event demos are '(off-list)'.

Output: configs/demo_list_final.csv  (one row per model demo).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
from reconcile_downloads import norm, event_tokens  # noqa: E402
from extract_demos import series_id  # noqa: E402

TD = ROOT / "data" / "training_dataset.parquet"
VAL = ROOT / "configs" / "parsed_demos_validation.csv"
CURATED = ROOT / "configs" / "inferno_matches_liquipedia.csv"
LOG = ROOT / ".cache" / "extract.log"
OUT = ROOT / "configs" / "demo_list_final.csv"

STAGE_KWS = ["closed-qualifier", "qualifier", "play-in", "opening-stage", "opening",
             "elimination", "stage-2", "stage-1", "stage-3", "groups", "group"]


def dem_to_rars() -> dict[str, list[str]]:
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
    d2r = dem_to_rars()
    by_rar = {r["rar_filename"]: r for r in curated if r.get("rar_filename")}
    by_sid = {}
    for r in curated:
        if r.get("rar_filename"):
            by_sid.setdefault(series_id(r["rar_filename"]), r)
    events = sorted({r["event"] for r in curated})

    def ctx(demo):  # source-archive context string (event + teams)
        if "__" in demo:
            return demo.split("__")[0]
        rs = d2r.get(demo + ".dem", [])
        return rs[0] if rs else ""

    def exact_row(demo):
        if "__" in demo:
            return by_sid.get(demo.split("__")[0])
        for rar in d2r.get(demo + ".dem", []):
            if rar in by_rar:
                return by_rar[rar]
            if series_id(rar) in by_sid:
                return by_sid[series_id(rar)]
        return None

    def event_from_ctx(c):
        cn = norm(c)
        best, k = None, -1
        for ev in events:
            toks = event_tokens(ev)
            if toks and all(t in cn for t in toks) and len(toks) > k:
                best, k = ev, len(toks)
        return best

    def stage_from_ctx(c):
        cl = c.lower()
        for kw in STAGE_KWS:
            if kw in cl:
                return kw.replace("-", " ")
        return "group"

    rows, off = [], 0
    for demo in sorted(model_ids):
        v = val.get(demo, {})
        clans = (v.get("clans") or "").split("/")
        base = {"demo_id": demo, "team_score": v.get("team_score", ""),
                "winner_team": v.get("winner_team", ""), "real_rounds": v.get("real_rounds", "")}
        row = exact_row(demo)
        if row:  # exact curated playoff match
            rows.append({**base, "event": row["event"], "stage": row["stage"],
                         "series_date": row["series_date"], "team_a": row["team_a"],
                         "team_b": row["team_b"], "hltv_match_url": row["hltv_match_url"]})
            continue
        ev = event_from_ctx(ctx(demo))
        ta, tb = (clans + ["?", "?"])[:2]
        if ev:  # Tier-1 game off the playoff list (group stage etc.)
            rows.append({**base, "event": ev, "stage": stage_from_ctx(ctx(demo)),
                         "series_date": "", "team_a": ta, "team_b": tb, "hltv_match_url": ""})
        else:
            off += 1
            rows.append({**base, "event": "(off-list)", "stage": "", "series_date": "",
                         "team_a": ta, "team_b": tb, "hltv_match_url": ""})

    out = pl.DataFrame(rows).select(
        ["demo_id", "event", "stage", "series_date", "team_a", "team_b",
         "team_score", "winner_team", "real_rounds", "hltv_match_url"]
    ).sort(["series_date", "event", "demo_id"])
    out.write_csv(OUT)
    print(f"WROTE {out.height} model demos -> {OUT}  ({off} truly off-list)")
    from collections import Counter
    for e, n in Counter(r["event"] for r in rows).most_common():
        print(f"  {n:3d}  {e}")


if __name__ == "__main__":
    main()
