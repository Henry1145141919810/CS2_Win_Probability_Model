# Map-Control Models (Pillar 2) — explained

Map control is our novel feature pillar: a number (per snapshot) for **how much of the map
each team dominates**. The hypothesis is that territorial dominance predicts round outcome
beyond economy/combat state. We implemented and compared **three** ways to measure it.

All three operate on the **nav mesh** — de_inferno is decomposed into **3,060 small walkable
"areas"** (polygons with a centroid). "CT control %" = the area-weighted fraction of those
areas owned by CT at a given instant. We use only **alive** players.

---

## 1. Voronoi control (`control_features` / `voronoi_owner`)

**Mechanism:** build a nearest-neighbor tree from the alive players' (x, y); assign **every
nav area to the team of the closest living player**. Area-weighted sum → CT control %.

```
for each nav area:
    owner = team of the nearest alive player        # pure proximity
ct_control = (area-weighted) fraction owned by CT
```

- It's a **hard partition**: every walkable cell belongs to whoever is closest — nothing is
  neutral, the two teams' shares always sum to 1.
- **Interprets control as proximity/positioning** — "who is physically closest to each region."
- **Strengths:** smooth, stable, robust; reflects overall map presence and where teams are
  pushing; doesn't depend on twitchy aim.
- **Weaknesses:** physically unrealistic — a player "controls" areas they cannot see (through
  walls) or are not looking at; it overstates control (each team always ~half the map).

We also derive: per-zone control (banana/mid/A/B/CT-spawn), 10-second **trend** (gaining or
losing territory), and **volatility**.

---

## 2. Grey / contestability control — instantaneous (`contest_control`)

**Mechanism:** an area is controlled by a team only if a living player can **actually contest
it right now**. A player contests an area if ALL hold:

1. **In range** (≤ ~1800 u), **or** immediately adjacent (≤ 280 u — you hold your own spot),
2. **Line of sight** — not through a wall (precomputed 3060×3060 visibility matrix, ray-cast
   against the map mesh; only **6.3 %** of area-pairs are mutually visible — Inferno is very
   walled),
3. **Facing it** — the area is within the player's field of view (from `yaw`),
4. **Not behind an active smoke**.

Each area is then one of **4 states**: **CT / T / contested (both) / grey (neither)**.

- **Captures the *real instantaneous* picture** — what each player can literally see/hold.
- ~**80 % of the map is grey** at any instant (players only watch a few angles).
- **Strengths:** physically faithful — respects walls, smokes, where players look.
- **Weaknesses:** **noisy and sparse**. Sightlines flicker tick-to-tick (crosshairs move
  constantly), so it's a snapshot of a fast-changing instant, not a stable state.

**Empirically this did NOT beat Voronoi** (it's noisier). FOV in particular added noise.

---

## 3. Territory control — with MEMORY + DECAY (`TerritoryControl`)  ← the current best idea

The fix for #2's flicker: **control you've established persists.** Once a team clears/sees an
area it becomes **their territory and stays theirs even after they look away**, until either
the enemy takes it or they neglect it too long.

**Mechanism** (stateful, processed in time order within a round):

```
per area, remember the last tick CT saw it (ct_seen) and T saw it (t_seen)
each snapshot:
    mark areas a team can currently see/hold (same gates as #2)  -> update *_seen = now
    an area is "CT-active"   if (now - ct_seen) <= decay   (decay = 15 s)
                "T-active"    if (now - t_seen)  <= decay
    state = CT / T / contested / grey  from the two *_active masks
```

- **FOV matters for *acquiring*** new or neglected space (re-peeking banana after 30 s),
  **not for *holding*** what you already cleared — exactly how real control works.
- A team can rotate away and keep its territory; if it ignores an area for >15 s, uncertainty
  grows and the area decays back to grey (the enemy could have repositioned).
- The enemy seeing an area immediately flips/contests it.

**Why it should be better:** it keeps the **realism** of #2 (walls, smokes, facing) but
restores the **stability** of #1 (held space persists), so it tracks the slow, outcome-relevant
state of the round instead of a flickering instant. On the banana round, CT territory builds to
0.53 then **decays to 0.07 as T executes** (T won) — a clean, interpretable signal.

`decay_sec` is the one tunable knob (default 15 s). The visibility matrix and smoke/facing
gates are shared with #2.

---

## How we judge them — and the key lesson

5-fold **GroupKFold** (by match), primary metric **AUC** (ranking: P(model scores a won
snapshot above a lost one)); plus Brier/log-loss (calibration), DeLong (significance), and
match-level **block bootstrap** (confidence intervals).

**Lesson so far:** *realistic ≠ predictive*. The round outcome is a slow/aggregate event, so a
**stable** control signal predicts it better than a physically-faithful but **flickering** one.
Voronoi (proximity) beat the instantaneous grey. The territory model (#3) is the attempt to get
both — realism **and** stability — and is the current frontier of the spatial pillar.
