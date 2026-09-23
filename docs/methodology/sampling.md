# Demo Sampling Protocol

The sampling frame for the training set. Decide *before* downloading so selection is
auditable and reportable (a reviewer will ask). Goal: **~150 clean `de_inferno` GOTV maps**
after filtering, drawn from Tier-1 pro CS2, Jan 2024 – Dec 2025.

> Test set (held out) = `de_inferno` demos from **Jan 2026 onwards**, downloaded separately
> at the end. Never mixed with training.

---

## 1. Inclusion / exclusion rules

**Include a map if all of:**
- Map is `de_inferno`.
- Match is from a **Tier-1 or strong Tier-2** event (see `configs/target_events.csv`).
- Series date is within **2024-01-01 … 2025-12-31**.
- GOTV (server-side) demo is available on HLTV.

**Exclude / discard:**
- Walkovers, forfeits, technical-loss maps.
- Overtime-heavy or aborted maps (parse may still keep them; flagged at validation).
- Maps with < 20 rounds after parsing.
- Online qualifiers / open-bracket matches (quality + roster-volatility noise).

## 2. Selection-bias safeguards (the part reviewers scrutinize)

| Rule | Value | Why |
|---|---|---|
| **Per-team cap** | ≤ 12 Inferno maps per team | Stops one team's setups dominating |
| **Per-event cap** | ≤ ~16 Inferno maps per event | Stops one meta/patch dominating |
| **Team breadth** | ≥ 20 distinct teams | Generalization across playstyles |
| **Skill spread** | mix Tier-1 vs strong Tier-2 | Gives firepower feature variance (ratings ~0.9–1.3) |
| **Side balance** | track CT vs T win rate | Inferno is historically CT-sided; don't let it skew the label |
| **Temporal spread** | roughly even across 2024 H1/H2, 2025 H1/H2 | Avoid era/meta overfit |

Log every collected map in `configs/inferno_matches_liquipedia.csv` so these can be checked and reported.

## 3. Target allocation

See `configs/target_events.csv`. Targets sum to **~210 Inferno maps** — deliberately
**more than 150** as buffer for filtering and for events where Inferno was picked less.
**Stop once ~150 clean maps are reached with good team/temporal spread**; the extra rows
are a prioritized backlog, not a quota.

Allocation priority order:
1. The three Majors (Copenhagen 2024, Shanghai 2024, Austin 2025) — most prestige, cleanest demos.
2. ESL Pro League seasons & IEM Katowice/Cologne — large, many Inferno maps.
3. BLAST finals, IEM Dallas/Chengdu — fill spread.

## 4. How to find the Inferno maps within an event (HLTV workflow)

HLTV has no global "filter results by map" button, so per event:

1. Go to the event page on HLTV (`hltv.org/events` → the event).
2. Open **"Maps"** / results; each series page lists the maps played with scores.
3. For each series that **played Inferno**, open the match page → **"GOTV Demo"** →
   download the `.rar` (one archive per series; contains a `.dem` per map).
4. Drop the `.rar` into `demos/raw/`. (Keep the whole archive — extraction is automated;
   non-Inferno maps in the archive are ignored at parse time.)
5. Update the matching row in `configs/inferno_matches_liquipedia.csv`
   (set `downloaded`, `rar_filename`).

Tip: HLTV's per-event **stats → maps** view, and Liquipedia event pages, both show which
maps each series played — faster than opening every match.

## 5. Priority matchups to check first

High-prestige series known to have occurred in this era (verify Inferno was actually played
on the match page before downloading — map-pick varies by series):

- **PGL Major Copenhagen 2024** (Mar 2024): playoffs incl. Natus Vincere, FaZe, Spirit, G2, Vitality, MOUZ, Heroic. Final: NaVi vs FaZe.
- **Perfect World Shanghai Major 2024** (Dec 2024): playoffs incl. Spirit, FaZe, NaVi, G2, MOUZ, Vitality, The MongolZ. Final: Spirit vs FaZe.
- **BLAST Austin Major 2025** (Jun 2025): playoffs incl. Vitality, MOUZ, The MongolZ, Spirit, FaZe, NaVi, Falcons, Aurora.
- **IEM Cologne 2024** (Aug 2024, won by Vitality), **IEM Katowice 2025** (Feb 2025, Vitality), **ESL Pro League S21** (Apr 2025, Vitality), **BLAST World Final 2024** (won by G2).

> These are starting points, not a verified Inferno list. The exact maps played per series
> must be confirmed on HLTV/Liquipedia. See open task to auto-extract a verified per-map list.

## 6. Team pool (Tier-1 / strong Tier-2, 2024–25 era)

Vitality, Natus Vincere (NaVi), G2, FaZe, Team Spirit, MOUZ, The MongolZ, Astralis,
Heroic, Team Liquid, Complexity, Cloud9, Eternal Fire, Falcons, Aurora, paiN, FURIA,
Virtus.pro, 3DMAX, GamerLegion, BIG, Ninjas in Pyjamas (NIP), ENCE, fnatic.
