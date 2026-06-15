"""Mine Liquipedia for de_inferno maps played at target Tier-1 CS2 events.

Builds a verified, navigable shopping list of Inferno maps (event, stage, date,
teams, score, Liquipedia link) so demos can be located and downloaded from HLTV.

Uses the Liquipedia MediaWiki API politely:
  - descriptive User-Agent (required by Liquipedia API ToS)
  - >= 2.5s between requests (parse-action rate limit is ~1 req / 2s)

Output: configs/inferno_matches_liquipedia.csv

Usage:
  python src/data/liquipedia_inferno.py            # all events
  python src/data/liquipedia_inferno.py --only PGL/2024/Copenhagen
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import re
import sys
import time
from pathlib import Path

import requests

# Liquipedia team-template aliases -> readable names (fallback: title-cased alias).
ALIASES = {
    "vit": "Vitality", "vitality": "Vitality", "navi": "Natus Vincere",
    "faze": "FaZe", "g2": "G2", "c9": "Cloud9", "mouz": "MOUZ",
    "vp": "Virtus.pro", "col": "Complexity", "pain": "paiN",
    "the mongolz": "The MongolZ", "mongolz": "The MongolZ",
    "eternal fire": "Eternal Fire", "furia": "FURIA", "spirit": "Team Spirit",
    "liquid": "Team Liquid", "astralis": "Astralis", "heroic": "Heroic",
    "falcons": "Falcons", "aurora": "Aurora", "ecstatic": "ECSTATIC",
    "imp": "Imperial", "3dmax": "3DMAX", "gamerlegion": "GamerLegion",
    "big": "BIG", "nip": "Ninjas in Pyjamas", "ence": "ENCE",
    "fnatic": "fnatic", "9z": "9z", "g2 esports": "G2",
}


def _name(alias: str) -> str:
    a = alias.strip()
    return ALIASES.get(a.lower(), a)

API = "https://liquipedia.net/counterstrike/api.php"
UA = {
    "User-Agent": "CS2WinProbResearch/0.1 (hyhuang@sas.upenn.edu) "
    "academic dataset build; contact for issues"
}
SLEEP = 12.0  # seconds between live API calls (Liquipedia is strict; be polite)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "inferno_matches_liquipedia.csv"
CACHE = ROOT / ".cache" / "liquipedia"

# Target events: label -> Liquipedia base page. Verified/guessed; the script
# reports which resolve. Qualifier subpages are filtered out automatically.
EVENTS = {
    "IEM Katowice 2024": "Intel Extreme Masters/2024/Katowice",
    "ESL Pro League Season 19": "ESL Pro League/Season 19",
    "PGL Major Copenhagen 2024": "PGL/2024/Copenhagen",
    "IEM Chengdu 2024": "Intel Extreme Masters/2024/Chengdu",
    "IEM Dallas 2024": "Intel Extreme Masters/2024/Dallas",
    "BLAST Premier Spring Final 2024": "BLAST/Premier/2024/Spring/Final",
    "IEM Cologne 2024": "Intel Extreme Masters/2024/Cologne",
    "ESL Pro League Season 20": "ESL Pro League/Season 20",
    "BLAST Premier Fall Final 2024": "BLAST/Premier/2024/Fall/Final",
    "Perfect World Shanghai Major 2024": "Perfect World/Major/2024/Shanghai",
    "BLAST Premier World Final 2024": "BLAST/Premier/2024/World Final",
    "IEM Katowice 2025": "Intel Extreme Masters/2025/Katowice",
    "ESL Pro League Season 21": "ESL Pro League/Season 21",
    "PGL Cluj-Napoca 2025": "PGL/2025/Cluj-Napoca",
    "IEM Dallas 2025": "Intel Extreme Masters/2025/Dallas",
    "BLAST Austin Major 2025": "BLAST/Major/2025/Austin",
    "IEM Cologne 2025": "Intel Extreme Masters/2025/Cologne",
    "ESL Pro League Season 22": "ESL Pro League/Season 22",
    "StarLadder Budapest Major 2025": "StarLadder/2025/Budapest",
    "BLAST Premier World Final 2025": "BLAST/Premier/2025/World Final",
}

# Subpage filtering: keep main-event stages, drop online/open qualifiers.
DROP = re.compile(
    r"Qualifier|/Open(/|$)|Americas|Europe|Asia-Pacific|Oceania|/China|/Decider|"
    r"Arabia|Africa|Levant|MENA|Closed|Last Chance",
    re.IGNORECASE,
)
KEEP = re.compile(
    r"Stage|Playoff|Group|Bracket|Legends|Challengers|Champions|Contenders|Main Event",
    re.IGNORECASE,
)

_session = requests.Session()
_session.headers.update(UA)


def api_get(params: dict, *, retries: int = 8) -> dict:
    """GET with 429/5xx backoff honoring Retry-After. Sleeps SLEEP after success."""
    params = {**params, "format": "json"}
    delay = 30.0
    for attempt in range(retries):
        r = _session.get(API, params=params, timeout=40)
        if r.status_code in (429, 502, 503):
            wait = max(int(r.headers.get("Retry-After", 0)), int(delay))
            print(f"  [rate-limited {r.status_code}] sleeping {wait}s "
                  f"(attempt {attempt+1}/{retries})", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 1.7, 180)
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    raise RuntimeError(f"giving up after {retries} retries: {params}")


def _cache_path(title: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", title)
    return CACHE / f"{safe}.wikitext"


def page_exists(title: str) -> bool:
    if _cache_path(title).exists():
        return True
    data = api_get({"action": "query", "titles": title, "prop": "info"})
    pages = data.get("query", {}).get("pages", {})
    return bool(pages) and all("missing" not in p for p in pages.values())


def list_subpages(base: str) -> list[str]:
    cp = CACHE / f"__subpages__{re.sub(r'[^A-Za-z0-9._-]', '_', base)}.txt"
    if cp.exists():
        return [ln for ln in cp.read_text(encoding="utf-8").splitlines() if ln]
    data = api_get(
        {
            "action": "query",
            "list": "allpages",
            "apprefix": base + "/",
            "apnamespace": "0",
            "aplimit": "200",
        }
    )
    titles = [p["title"] for p in data.get("query", {}).get("allpages", [])]
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text("\n".join(titles), encoding="utf-8")
    return titles


def get_wikitext(title: str) -> str:
    cp = _cache_path(title)
    if cp.exists():
        return cp.read_text(encoding="utf-8")
    data = api_get({"action": "parse", "page": title, "prop": "wikitext"})
    wt = data.get("parse", {}).get("wikitext", {}).get("*", "")
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(wt, encoding="utf-8")
    return wt


def search_pages(label: str, limit: int = 6) -> list[str]:
    data = api_get(
        {"action": "query", "list": "search", "srsearch": label, "srlimit": str(limit)}
    )
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def extract_template_blocks(text: str, name: str) -> list[str]:
    """Return balanced {{name...}} substrings, ensuring the char after `name`
    is not alphanumeric (so {{Match}} != {{Matchlist}})."""
    out, pos, token = [], 0, "{{" + name
    n = len(text)
    while True:
        s = text.find(token, pos)
        if s < 0:
            break
        after = text[s + len(token)] if s + len(token) < n else ""
        if after.isalnum():
            pos = s + len(token)
            continue
        depth, i = 0, s
        while i < n:
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
                if depth == 0:
                    break
            else:
                i += 1
        out.append(text[s:i])
        pos = i
    return out


def parse_opponents(block: str) -> list[str]:
    names = re.findall(r"\{\{[0-9]*TeamOpponent\|([^|}\n]+)", block)
    if len(names) < 2:
        names = re.findall(r"\|opponent[12]\s*=\s*\{\{[^|]*\|([^|}\n]+)", block)
    return [_name(n) for n in names[:2]]


def _half(block: str, key: str) -> int:
    return sum(int(x) for x in re.findall(rf"\|\s*{key}\s*=\s*(\d+)", block))


def parse_inferno_score(block: str):
    """Find the {{Map}} subtemplate whose map=Inferno and was actually played;
    return (score_a, score_b) as strings, or None if not played."""
    for m in extract_template_blocks(block, "Map"):
        if not re.search(r"\|\s*map\s*=\s*inferno\b", m, re.IGNORECASE):
            continue
        if re.search(r"\|\s*finished\s*=\s*skip", m, re.IGNORECASE):
            return None  # picked but not played -> no demo
        # explicit score first, else sum CT/T (and OT) halves
        s1 = re.search(r"\|\s*score1\s*=\s*(\d+)", m)
        s2 = re.search(r"\|\s*score2\s*=\s*(\d+)", m)
        if s1 and s2:
            a, b = int(s1.group(1)), int(s2.group(1))
        else:
            a = _half(m, "t1t") + _half(m, "t1ct") + _half(m, "t1ot")
            b = _half(m, "t2t") + _half(m, "t2ct") + _half(m, "t2ot")
        if a == 0 and b == 0:
            return None  # not actually played
        return (str(a), str(b))
    return None


_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def parse_date(block: str) -> str:
    m = re.search(r"\|\s*date\s*=\s*(\d{4}-\d{2}-\d{2})", block)
    if m:
        return m.group(1)
    m = re.search(r"\|\s*date\s*=\s*([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", block)
    if m and m.group(1) in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
    return ""


def parse_hltv(block: str) -> str:
    m = re.search(r"\|\s*hltv\s*=\s*(\d+)", block)
    return m.group(1) if m else ""


def parse_bestof(block: str) -> str:
    m = re.search(r"\|\s*bestof\s*=\s*(\d+)", block)
    return m.group(1) if m else ""


def stage_from_title(base: str, title: str) -> str:
    if title == base:
        return "Main/Playoffs"
    return title[len(base) + 1 :]


def resolve_base(label: str, base: str) -> str | None:
    """Return a working Liquipedia base page for `label`. If the configured `base`
    is missing, search Liquipedia and pick the first candidate whose wikitext looks
    like a tournament page (has Match templates or a tier field)."""
    if get_wikitext(base):
        return base
    for cand in search_pages(label, limit=6):
        wt = get_wikitext(cand)
        if wt and ("{{Match" in wt or "liquipediatier" in wt):
            print(f"  [resolved] '{base}' -> '{cand}'", file=sys.stderr)
            return cand
    return None


def mine_event(label: str, base: str) -> list[dict]:
    resolved = resolve_base(label, base)
    if not resolved:
        print(f"  [SKIP] unresolved: {base}", file=sys.stderr)
        return []
    base = resolved
    titles = [base]
    for sub in list_subpages(base):
        if DROP.search(sub):
            continue
        if KEEP.search(sub):
            titles.append(sub)
    print(f"  pages to scan ({len(titles)}): {[t[len(base):] or '/' for t in titles]}")
    rows = []
    for title in titles:
        try:
            wt = get_wikitext(title)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] fetch failed {title}: {e}", file=sys.stderr)
            continue
        stage = stage_from_title(base, title)
        for block in extract_template_blocks(wt, "Match"):
            score = parse_inferno_score(block)
            if score is None:
                continue
            opps = parse_opponents(block)
            if len(opps) < 2:
                continue
            hltv_id = parse_hltv(block)
            rows.append(
                {
                    "event": label,
                    "stage": stage,
                    "series_date": parse_date(block),
                    "team_a": opps[0],
                    "team_b": opps[1],
                    "bo_format": parse_bestof(block),
                    "inferno_score_a": score[0],
                    "inferno_score_b": score[1],
                    "hltv_match_url": f"https://www.hltv.org/matches/{hltv_id}/-"
                    if hltv_id else "",
                    "liquipedia_page": "https://liquipedia.net/counterstrike/"
                    + title.replace(" ", "_"),
                    # --- download tracking (filled in as you download/parse) ---
                    "rar_filename": "",
                    "downloaded": "",
                    "parsed": "",
                    "n_rounds": "",
                    "notes": "",
                }
            )
    # dedup on (event, teams, date)
    seen, uniq = set(), []
    for r in rows:
        key = (r["event"], frozenset((r["team_a"].lower(), r["team_b"].lower())), r["series_date"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    print(f"  -> {len(uniq)} Inferno map(s)")
    return uniq


def resolve_mode():
    """Print Liquipedia search candidates for each event so page names can be fixed."""
    for label, base in EVENTS.items():
        ok = page_exists(base)
        print(f"\n[{label}]  configured='{base}'  exists={ok}")
        if not ok:
            for t in search_pages(label):
                print(f"    candidate: {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single Liquipedia base page")
    ap.add_argument("--resolve", action="store_true",
                    help="search for correct page names of unresolved events")
    args = ap.parse_args()

    if args.resolve:
        resolve_mode()
        return

    events = EVENTS
    if args.only:
        label = next((k for k, v in EVENTS.items() if v == args.only), args.only)
        events = {label: args.only}

    all_rows = []
    try:
        for label, base in events.items():
            print(f"[{label}]  {base}")
            try:
                all_rows.extend(mine_event(label, base))
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] {label}: {e}", file=sys.stderr)
    finally:
        _write(all_rows)


def _write(all_rows):

    cols = [
        "event", "stage", "series_date", "team_a", "team_b", "bo_format",
        "inferno_score_a", "inferno_score_b", "hltv_match_url", "liquipedia_page",
        "rar_filename", "downloaded", "parsed", "n_rounds", "notes",
    ]
    track_cols = ["rar_filename", "downloaded", "parsed", "n_rounds", "notes"]

    def row_key(r):
        return (
            r["event"],
            frozenset((r["team_a"].lower(), r["team_b"].lower())),
            r["series_date"],
        )

    # Merge: preserve any tracking values already present in the existing file.
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            prev = {row_key(r): r for r in csv.DictReader(f)}
        carried = 0
        for r in all_rows:
            old = prev.get(row_key(r))
            if old:
                for c in track_cols:
                    if old.get(c):
                        r[c] = old[c]
                        carried = carried or True
        if carried:
            print("  (preserved existing download-tracking values where present)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWROTE {len(all_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
