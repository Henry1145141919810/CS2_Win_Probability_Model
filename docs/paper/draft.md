# Where Win Probability Comes From — and Where It Breaks

### Spatial Control, Bomb Geometry, and an Out-of-Time Reality Check for Live Win Probability in Counter-Strike 2

**Henry Huang¹, Leu², Abraham J. Wyner¹**
¹ Wharton Sports Analytics & Business Initiative, University of Pennsylvania · ² *(affiliation TBD)*

**Target:** MLSA 2027 (ECML/PKDD workshop), 9 pages Springer LNCS · arXiv preprint first
**Status:** FIRST DRAFT (2026-07-03). Numbers are final unless marked ⚠️.

---

## Abstract

Live win-probability models are now standard broadcast furniture in esports, yet the published
Counter-Strike literature evaluates them almost exclusively on economy and combat state, pooled over
all game situations, and validated only by cross-validation. We build a per-second win-probability
model for Counter-Strike 2 (map: de_inferno) from 220 professional matches (476,595 per-second
snapshots) and use it to ask three questions the pooled-AUC convention obscures: *where* does the
signal actually live, *which* spatial representation is predictive, and *does any of it survive
deployment?*

We contribute four findings. **(1)** The conventional headline AUC (≈0.85) is carried almost entirely
by lopsided game states; restricted to genuinely even rounds — equal players alive and equal
equipment — every model falls to ≈0.58, barely above a coin flip. We formalise this as
**contested-AUC** and argue it, not pooled AUC, is the metric a live model should be judged on.
**(2)** Adapting Voronoi pitch control from football, we compare three increasingly *physically
faithful* map-control representations and find that faithfulness and predictiveness diverge: an
instantaneous line-of-sight/field-of-view/smoke model is the most realistic and the *least*
predictive, and only regains parity with naive proximity control once it is given temporal memory.
Territorial control is predictive because it is *stable*, not because it is *correct*.
**(3)** A physically-motivated **defuse-race** feature — fuse time remaining minus the nearest
counter-terrorist's path time minus the defuse duration — is the single strongest addition beyond
economy, cutting post-plant log-loss by 7–8%. **(4)** Across nine architectures (logistic regression,
random forest, XGBoost, LightGBM, CatBoost, a causal TCN, a causal Transformer, a player-graph
attention network, and an ensemble), with match-level block-bootstrap confidence intervals, *all* are
statistically indistinguishable: sequence and graph deep models learn nothing that careful feature
engineering has not already captured at this data scale.

Finally, we subject the model to a **touch-once out-of-time holdout** on 27 unseen 2026 matches. The
demo-derived pillars — economy, map control, bomb geometry — transfer essentially perfectly
(0.8493 → 0.8501). But the model's best in-sample configuration becomes its *worst* out-of-time
(0.8519 → 0.8236, ECE 0.016 → 0.071), because its player-skill pillar carries an **inference-time data
dependency**: it requires an external, per-era player database that the deployment era does not yet
have. We argue this failure mode — invisible to cross-validation by construction — is a general hazard
for any "prior" feature sourced outside the observation itself.

---

## 1. Introduction

**Motivation.** Win probability is the interpretive backbone of live sports coverage: it converts a
complex state into one number a viewer can act on. In esports the state is unusually rich — every
player's position, velocity, view angle, health, equipment, and utility are recorded at 64 Hz — and
yet published Counter-Strike win-probability models remain dominated by *economy* features
(equipment value, players alive, bomb status). The spatial half of the game, which players and
analysts talk about constantly ("we lost map control", "they're stacked B"), is largely unmodelled.

Three methodological habits in this literature obscure how good these models really are:

1. **Pooled evaluation.** A single AUC over all snapshots mixes 5-v-1 blowouts with 5-v-5 stalemates.
   A model that only knows "who has more players and money" scores well because most snapshots are
   easy.
2. **Faithfulness assumed to imply predictiveness.** Richer spatial models (line-of-sight, vision
   cones, smoke occlusion) are assumed superior because they are physically truer.
3. **Cross-validation as the final gate.** Random or grouped folds sample from the same era as
   training; they cannot see failures that appear only when the world moves on.

**This paper.** We build a per-second CS2 win-probability model on de_inferno and attack all three
habits. Our contributions:

- **C1 — Contested-AUC.** A conditional evaluation lens. Restricting to snapshots with equal players
  alive *and* near-equal equipment, the economy baseline falls from 0.846 to **0.580**. Pooled AUC
  systematically flatters win-probability models; contested-AUC is the honest measure (Fig. 2).
- **C2 — Three map-control representations, and a negative result.** Voronoi proximity control
  (adapted from football's pitch control), an instantaneous line-of-sight/FOV/smoke "grey" model, and
  a memory-and-decay "territory" model. The most faithful model is the least predictive; adding
  temporal memory restores it. **Predictiveness comes from *stability*, not fidelity.**
- **C3 — Defuse-race geometry.** A single physically-derived feature (fuse-time − path-time − defuse-time)
  is the strongest post-economy addition, cutting post-plant log-loss 7–8% while *preserving* calibration.
- **C4 — Nine architectures, one dead heat, and an out-of-time reality check.** With match-level
  bootstrap CIs, sequence (TCN, Transformer) and graph (GAT) deep models are statistically tied with
  a calibrated logistic regression (Fig. 1). And on a touch-once 2026 holdout, the *skill-prior*
  pillar collapses due to an inference-time data dependency, while the demo-derived pillars transfer
  intact (Fig. 4). We believe C4 is the most transferable lesson of the paper.

---

## 2. Related Work

**Counter-Strike win probability.** Xenopoulos et al. introduce ESTA and a per-round win-probability
baseline built on economy/combat state, establishing the convention we adopt as our Model A and
extend. Community models (e.g. `andrewq-qiu/cs2-win-probabilities`) predict at round granularity from
a frozen post-buy snapshot; we instead predict at every second of the round, which is what a live
broadcast overlay requires. ⚠️ *TODO: full citations; check for 2025–26 CS2-era follow-ups.*

**Spatial control in team sports.** Voronoi-based *pitch control* is well established in football
tracking analytics (Taki & Hasegawa; Fernández & Bornn; Spearman; the Friends-of-Tracking line of
work). Its transfer to a tactical shooter is not obvious: football pitch control is governed by
reachability under motion models, whereas in CS2 "control" is mediated by sightlines, weapon range,
utility (smokes), and — as we show — *memory* of previously cleared space.

**Deep models on tracking data.** Graph attention over player nodes and temporal convolutions over
state sequences are the standard architectures for tracking-derived prediction. Our contribution is
negative but carefully quantified: at 220 matches, neither beats a linear model with good features.

**Calibration and evaluation.** We adopt proper scoring rules (log-loss, Brier), the Murphy
decomposition, calibration slope/intercept (the TRIPOD convention in clinical prediction), and
binning-free calibration estimates. Choi et al. argue for replacing accuracy with calibration-aware
metrics specifically in esports win probability, which motivates our extended battery.

---

## 3. Data

**Source and scope.** 220 professional de_inferno matches (2024–2025, Tier-1 events), parsed with
`awpy` v2.0.2 at a forced 64-tick rate. We restrict to a single map so that spatial features are
directly comparable across matches; generalising across maps is left to future work (§9).

**Snapshots.** Each round is sampled **once per second** from freeze-end to round end. Every snapshot
is one training row, labelled with the round's eventual winner (`ct_won`). This yields **476,595
snapshots**, base rate P(CT win) = **0.445**. Snapshots within a round share a label; the probability
moves because the *state* moves.

**Validation.** Rounds are reconstructed defensively: side assignment is recovered via team-clan
mode (guarding against the halftime swap), and clinch-trimmed rounds are dropped. 44 of 264 parsed
demos fail validation and are excluded.

**Out-of-time test set.** A separate bundle of **27 unseen 2026 matches** (55,271 snapshots; IEM
Kraków / Cologne Major / Atlanta, BLAST Rivals) — zero overlap with training — is held out and
touched **exactly once**, at the end (§7.8). Its base rate is **0.512**: 2026 Inferno is measurably
more CT-sided, itself a distribution shift we must handle.

---

## 4. Feature Pillars

We organise features into four pillars. Set names used throughout: **A** = economy; **E** = A + map
control + tactical/bomb; **EB2** = E + defuse-race; **F** = A + firepower; **EF/EFB2** = combinations.

### 4.1 Pillar 1 — Economy and combat (17 features)
The literature baseline: equipment value, health, armour, players alive, defuse kits, score, time
elapsed, bomb-planted flag. Aggregated over **alive** players only.

### 4.2 Pillar 2 — Map control, three ways (the ablation)

We adapt Voronoi pitch control to the de_inferno navigation mesh (3,060 areas). For each snapshot we
compute, for every nav area, which team "controls" it, then aggregate to area-weighted control
fractions (overall and per macro-zone: A site, B site, banana, mid, CT spawn), plus a rolling trend
and volatility.

- **(a) Voronoi (proximity).** Each nav area is assigned to the nearest *living* player's team via a
  KD-tree. Crude — it ignores walls, vision, and smokes.
- **(b) Grey (instantaneous LOS + FOV + smoke).** An area is controlled only if a living player is in
  range **and** has line-of-sight (a precomputed 3060×3060 nav visibility matrix, ray-cast against the
  map mesh) **and** is facing it (field-of-view from yaw) **and** it is not occluded by an active
  smoke. A four-state surface (CT / T / contested / grey). This is by far the most physically faithful
  model: ≈60–80% of the map is "grey" (uncontested) at any instant, which matches how CS actually plays.
- **(c) Territory (grey + memory and decay).** The grey model, but *cleared space stays yours*: an area
  marked by a team remains theirs for `decay` = 15 s without re-checking; FOV only gates *re-acquiring*
  neglected space; a player always holds his immediate surroundings. Stateful within a round.

The result (§7.3) is the paper's cleanest conceptual finding: **(b) is the most realistic and the least
predictive; (c) restores it.** All three are retained in the dataset for the ablation.

### 4.3 Pillar 3 — Tactical readiness and bomb geometry (42 features)

Utility inventories, AWP presence and zone, per-zone player counts, positional (Shannon) entropy, and
bomb geometry: plant site and coordinates, nearest-CT straight-line and **nav-mesh Dijkstra path**
distance to the bomb.

**Defuse-race (the key feature).** Post-plant, the round is a race. We track the bomb's live state
(*carried / dropped / planted*) from the event stream and compute, for each alive CT, an arrival time
(nav path ÷ run speed) plus a defuse duration (5 s with a kit, 10 s without); the defusing CT is the
one with the smallest finish time. Then

> **`defuse_time_margin` = fuse time remaining − (path time + defuse time)**

is positive iff a defuse is physically attemptable. A refinement (`defuse_margin_kit`, per-CT
kit-aware, plus a T-interruption window) is also computed. §7.4 shows this is the strongest
post-economy feature in the study.

We also tested **map control *around* the live bomb** and a **dropped-bomb scramble** feature. Both
were weak/redundant — reported as negative results (§7.4).

### 4.4 Pillar 4 — Firepower (player skill prior)

A pre-round skill prior from HLTV per-player statistics, joined on `(steamid, year)` because skill
drifts season to season. **v1** summed Rating/ADR/KAST over alive players plus a 1-v-N clutch score
(9 features). **v2** is side-aware (CT-side vs T-side Rating/Firepower/Entrying/Trading/Opening) with
per-player situational gates (lone survivor → clutch; teammates alive → entry/trading; opening only at
5-v-5), an AWP-holder sniping role flag, and grenade-value-weighted utility (20 features).

**This pillar is structurally different from the other three**, in a way that turns out to matter
enormously (§7.8): pillars 1–3 are computed *from the demo itself* and are therefore always available
at inference time. Pillar 4 requires an **external database**, indexed by era.

---

## 5. Models

Nine architectures, all evaluated identically.

**Classical (5).** Logistic regression (standardised, L2); random forest (300 trees); XGBoost,
LightGBM, CatBoost — all tuned to shallow, regularised trees (depth 3, λ = 10, subsample 0.8), which
fixed a depth-6 overfitting pathology worth +0.007 AUC.

**Deep (3).** All trained on GPU (NVIDIA B200, PARCC Betty).
- **Causal TCN** — the round as a sequence of per-second feature vectors; dilated causal convolutions
  (dilations 1/2/4), so the prediction at second *t* attends only to *t′ ≤ t* (no within-round
  look-ahead). Sequence-to-sequence: one probability per second.
- **Causal Transformer encoder** — same input, causal self-attention mask, sinusoidal positions.
- **GAT (player graph)** — consumes the **raw per-player trajectory**: position, velocity, view angle,
  health, armour, equipment, kit, side, for up to 10 alive players. Masked multi-head self-attention
  over the player set (a GAT on the complete player graph), permutation-invariant, masked mean-pooled.
  This is the only model that sees the game as *players in space* rather than as aggregates.

**Ensemble (1).** Soft-vote (mean probability) over classical + deep members. A logistic stack was also
tried and rejected (§7.6).

---

## 6. Evaluation Protocol

**Cross-validation.** 5-fold **GroupKFold by match**. Never split a match: rounds within a match are
correlated, and folding by round leaks and inflates AUC. All in-time numbers are out-of-fold (OOF).

**Metrics.**
- *Primary (literature-comparable):* AUC, log-loss, Brier.
- *Complementary:* ECE (10-bin), Brier Skill Score, and **contested-AUC** — AUC restricted to
  snapshots with equal players alive **and** |equipment difference| ≤ \$1500 (12.2% of snapshots).
- *Extended:* Murphy decomposition (reliability / resolution / uncertainty), sharpness,
  **calibration slope and intercept** (bin-free), adaptive equal-mass ECE and a KS calibration error
  (binning-free), and a **comeback/tail** diagnostic (§7.7).

**Uncertainty.** **Match-level block bootstrap** (B = 500): matches are the independent resampling
unit; each iteration draws all matches with replacement and recomputes *every* metric on the fixed OOF
predictions. This yields 95% CIs on AUC, log-loss, Brier, ECE, BSS and contested-AUC alike. DeLong's
test corroborates AUC differences. Deep models additionally report a multi-seed mean ± sd.

**Out-of-time.** A **touch-once** 2026 holdout (§7.8). We ran it exactly once; no model, feature, or
hyperparameter decision was made after seeing it.

---

## 7. Results

### 7.1 The model matrix

*(Fig. 3 — model × feature-set heatmap; Fig. 1 — forest plot with CIs)*

In-sample, the best single model is **logistic regression on all pillars (EFB2): AUC 0.8519**
(95% CI 0.8451–0.8582), and the best overall is the **4-model soft-vote ensemble: 0.8531**
(0.8460–0.8598), which also has the best calibration (ECE 0.009, BSS 0.378). The economy baseline is
**0.8465**. Every pillar's lift is small in pooled AUC but statistically significant (bootstrap CIs
exclude zero).

That logistic regression *wins* is itself informative: the aggregate signal is close to linear.

### 7.2 Where the signal actually lives — contested-AUC

*(Fig. 2)*

Pooled AUC hides the model's real behaviour. Conditioning on how *even* the round is:

| subset | share | AUC (A: economy) | AUC (EB2) |
|---|---|---|---|
| All snapshots | 100% | 0.844 | 0.849 |
| Equal players alive | 52% | 0.705 | 0.711 |
| Even economy | — | 0.683 | 0.689 |
| **Contested (both)** | **12.2%** | **0.580** | **0.586** |

**The economy baseline collapses from 0.844 to 0.580 — barely above a coin flip.** The headline number
is carried by lopsided snapshots (5-v-2, eco vs full-buy) that any model gets right. In the states a
viewer actually cares about — the round is even, the outcome is live — *no model in this study exceeds
0.60*. We regard this as the honest characterisation of the state of the art, and contested-AUC as the
metric future work should report.

The spatial pillars provide a consistent lift *throughout*, slightly larger in the contested regime.
Their aggregate contribution is modest (≈ +0.003–0.005 AUC) but robust: significant across all five
classical architectures.

### 7.3 Map control: realistic ≠ predictive (unless stabilised)

The central spatial result. XGBoost, in-time OOF:

| representation | AUC | verdict |
|---|---|---|
| Economy only (A) | 0.8443 | baseline |
| **+ Voronoi (proximity)** | **0.8468** | **best single control model** |
| + Grey (LOS + FOV + smoke) | 0.8451 | *most faithful, least predictive* |
| + Territory (grey + 15 s memory) | 0.8453 | memory restores it |

The instantaneous LOS/FOV/smoke model is a far better description of what the players can actually
*see and contest* — and it is a **worse** predictor than naive proximity. The reason is temporal: yaw is
twitchy and the visible set churns tick to tick, so instantaneous control **flickers**, while round
outcomes depend on *stable* territorial state. Granting the same model a 15-second memory (cleared
space stays yours) recovers parity with Voronoi.

> **Predictiveness comes from temporal stability, not physical fidelity.** We suspect this generalises
> to other tracking domains where "control" is defined instantaneously.

Voronoi and territory are also largely redundant with one another (r = 0.47; adding territory on top of
Voronoi yields ≈ 0); we retain Voronoi as the primary control feature.

### 7.4 Bomb geometry: the defuse race

`defuse_time_margin` is **permutation-importance rank #8 of 68** — the strongest feature added after
economy. Adding the bomb-live pillar (set EB2) is significant on **every** architecture (all bootstrap
CIs exclude 0) and is concentrated exactly where it should be:

- **post-plant log-loss −7–8%** (XGBoost 0.187 → 0.172; AUC 0.962 → 0.967),
- **endgame log-loss** falls across all models (0.401 → 0.396),
- **calibration is preserved** (ECE ≈ 0.013).

SHAP dependence shows the mechanism: the feature's contribution swings sharply across the
*feasibility boundary* at zero — when a defuse becomes physically impossible, the model correctly and
confidently abandons the CTs.

**Negative results (reported):** map control *around* the live bomb, and the dropped-bomb scramble,
were both weak/redundant. A loose bomb is a *consequence* of a losing round, not a cause; and
bomb-neighbourhood ownership duplicates existing site-control features. What matters post-plant is
the **race geometry**, not the local ownership.

### 7.5 Firepower: a pillar with a confound (in-sample)

Firepower is the weakest pillar. In-sample, F − A is significant only on logistic regression
(+0.0022, CI 0.0003–0.0045); on XGBoost/LightGBM/CatBoost the CI **includes zero**. Adding firepower on
top of the other pillars (EF − E) is ≈ 0 on every model, and *significantly negative* on CatBoost
(−0.0026) in v2, whose sparse, situationally-gated features the tree models overfit.

**A confound we flag explicitly.** `ct_rating_sum − t_rating_sum` is permutation-importance rank #1 —
but it correlates **0.987** with the *player-count advantage*, and predicts the outcome identically
(both r = 0.498). Because rating is **summed over alive players**, the feature is largely a re-encoding
of "who has more players alive", which economy already supplies. Its apparent dominance is an artifact;
its marginal contribution is ≈ 0. We report this rather than repair it, because §7.8 shows the pillar's
binding constraint is elsewhere.

### 7.6 Deep models: a statistical dead heat

*(Fig. 1)*

| model | AUC | 95% CI |
|---|---|---|
| Ensemble (soft-vote) | 0.8531 | (0.8460, 0.8598) |
| Logistic (EFB2) | 0.8519 | (0.8451, 0.8582) |
| LightGBM (EB2) | 0.8493 | (0.8424, 0.8558) |
| **TCN** (causal, sequence) | 0.8488 | (0.8420, 0.8557) |
| **Transformer** (causal) | 0.8473 | (0.8398, 0.8541) |
| **GAT** (raw trajectories) | 0.8465 | (0.8396, 0.8534) |

**Every point estimate lies inside the others' 95% CIs.** Neither the sequence models (which can see
*momentum*) nor the graph model (which sees the raw positions, velocities and view angles that the
aggregates throw away) beats a calibrated logistic regression.

Three observations. **(i)** Sequence length is critical for the TCN (AUC 0.849 at 160 steps, 0.826 at
100, 0.780 at 64) — truncation discards the decisive late-round snapshots. **(ii)** Multi-seed variance
is tiny (0.8485 ± 0.0004): the tie is not an artifact of unlucky initialisation. **(iii)** The deep
models nevertheless **earn their place in the ensemble**: the soft-vote of classical + deep beats either
family alone, which is the textbook diversity benefit. A logistic *stack* matched its AUC but destroyed
calibration (ECE 0.042) and was rejected.

**Interpretation.** The engineered features already encode what these architectures would have to learn
(trend/volatility for the TCN; zone control for the GAT), and 220 matches (≈4,900 rounds) is small for
a neural network. This is a statement about *this data scale*, not about deep learning.

### 7.7 Honest probabilities: calibration, and comebacks

All non-RF models are well calibrated (ECE < 0.02, robust to binning; calibration slope ≈ 1.0). The
Murphy decomposition shows the gain from adding pillars is **resolution** (discrimination), not
reliability: the features add skill *without* costing calibration.

We also introduce a **comeback/tail** diagnostic, since standard metrics say nothing about whether a
live model's *write-offs* are trustworthy. On the tails, the model says 5% → the side wins 0.7% of the
time; says 10% → 6.8%. The eventual winner was written off (≤ 10%) at some moment in **7.2% of rounds**,
and those calls are calibrated. A viewer watching the curve dip to 10% and seeing a comeback ~1 time in
14 is seeing an honest number.

### 7.8 ⭐ The out-of-time holdout: the model transfers; its skill prior does not

*(Fig. 4; holdout figure)*

We trained on all 220 matches (2024–25) and evaluated **once** on the 27 unseen 2026 matches.

**The demo-derived pillars transfer essentially perfectly:**

| set | model | in-time | out-of-time (2026) | Δ |
|---|---|---|---|---|
| **EB2** | LightGBM | 0.8493 | **0.8501** | **+0.0008** |
| EB2 | XGBoost | 0.8489 | 0.8497 | +0.0007 |
| E | XGBoost | 0.8476 | 0.8476 | +0.0000 |

Contested-AUC even *improves* out-of-time (0.590 → 0.64–0.65). Economy, Voronoi map control, tactical
readiness and defuse-race geometry all generalise to a new season, new patches, and new rosters.

**But the best in-sample model becomes the worst out-of-time:**

| model | in-time | out-of-time | Δ |
|---|---|---|---|
| **Logistic EFB2** | **0.8519** (best in-sample) | **0.8236** | **−0.0283** |
| Random forest EFB2 | 0.8446 | 0.8159 | −0.0287 |

Calibration breaks with it: ECE **0.016 → 0.071**, calibration intercept **−0.36** (vs a benign +0.11
for the non-firepower sets, which is just the base-rate shift). The probabilities become untrustworthy —
for a live product, the more serious failure of the two.

**Diagnosis (Fig. 4).** It is a *data-coverage* failure, not a signal failure. In 2026, **≈30% of
players have no entry in the skill database**; their contribution silently becomes zero. The team
skill-sum feature, tightly clustered at 5.28 in training (all five players known), collapses to a mean
of 3.66 with a long tail toward zero on the holdout. **The same feature means something different in
2026.** The model reads the corrupted input as "few/weak players alive" and mispredicts systematically.

**The lesson.** Pillars 1–3 are computed *from the observation itself* and are therefore always
available. Pillar 4 is a **prior sourced from outside the observation**, and therefore carries an
**inference-time data dependency**: at deployment, it needs an external database, indexed by an era
that — by definition — is the one you have least data for. When that database lags, the feature does
not merely stop helping: **it actively harms the model, and silently.**

Cross-validation cannot see this. Every fold is drawn from the same era, where the database is
complete. **The failure is invisible by construction to the standard protocol.** We therefore argue
that any model incorporating externally-sourced priors requires an out-of-time gate, and that its
production configuration should be chosen on out-of-time evidence. **Ours is EB2 — with no firepower —
even though cross-validation ranked EFB2 first.**

⚠️ *In progress: the 2026 skill data does exist and is being collected. We will report a second,
separately disclosed evaluation with (a) same-era statistics and (b) a leak-free lagged prior (previous
season's statistics), which is the deployment-realistic construction. Note that same-era statistics are
themselves mildly leaky — a player's 2026 rating is computed partly from the very matches being
predicted, and would not even exist at live inference time.*

---

## 8. Discussion

**Report contested-AUC.** Pooled AUC on a per-second win-probability model is close to a measure of
how often the game is lopsided. Any future CS win-probability work should report performance on the
even-round subset, where all current models are near chance and where improvement actually matters.

**Stability beats fidelity.** The grey-vs-territory result suggests a general principle for tracking
data: when the target is a *future* outcome, a representation's *temporal stability* can matter more
than its instantaneous correctness. Practitioners building vision- or reachability-based control
surfaces should test a memory-augmented variant before concluding that control "doesn't help".

**Physically-derived features beat learned ones at this scale.** The defuse-race margin — three
quantities and a subtraction — outperformed everything the deep models discovered on their own. With
~5k rounds, encoding known physics is a better use of capacity than learning it.

**Beware the external prior.** We think this is the most portable finding. A "skill prior", "player
form", "team strength" or any feature joined from an outside table introduces a dependency that
cross-validation is blind to. It should be (a) gated out-of-time, (b) constructed as a **lag** (only
information available before the event), and (c) monitored for coverage in production, with an explicit
fallback when coverage drops.

---

## 9. Limitations

1. **Single map.** All results are de_inferno. Spatial features are map-specific by construction;
   multi-map generalisation is untested.
2. **Scale.** 220 matches / ≈4,900 rounds. The deep-model dead heat is a statement about this regime;
   we would expect the GAT in particular to benefit from substantially more data.
3. **Firepower leakage.** Same-era player ratings are partly computed from the matches being
   predicted. We flag this; the lagged-prior variant (in progress) is the clean construction.
4. **Firepower count confound.** The summed-rating feature is ≈ a player-count proxy (r = 0.987). We
   report it rather than repair it; a per-capita encoding was considered and not pursued, since §7.8
   showed the pillar's binding constraint is data coverage, not encoding.
5. **Irreducible ceiling.** Contested rounds sit near 0.58 for *every* model tested. Some of this is
   genuine aleatoric randomness in an even CS round; how much is unknown.
6. **Deep-model holdout.** The deep models were not re-evaluated out-of-time (they inherit the same
   firepower dependency). ⚠️

---

## 10. Conclusion

We built a per-second win-probability model for CS2 and used it to interrogate three conventions of the
field. Pooled AUC flatters: restricted to even rounds, every model is near a coin flip, and we propose
contested-AUC as the honest metric. Physical fidelity does not imply predictiveness: the most realistic
map-control model is the least useful until it is given memory. And cross-validation is not a
deployment test: our best cross-validated model became our worst out-of-time, because one pillar
depended on a database the future does not yet have — a failure mode that grouped folds cannot expose.

The features that *did* transfer — economy, Voronoi map control, and a three-term defuse-race
inequality — are the ones computed directly from the game itself.

---

## Figures

| # | file | caption |
|---|---|---|
| **1** | `F1_forest.png` | All nine architectures with 95% match-level bootstrap CIs. Every point estimate lies inside the others' intervals: sequence and graph deep models are statistically tied with a calibrated logistic regression. |
| **2** | `F2_collapse.png` | AUC by round subset. The headline AUC is carried by lopsided snapshots; in genuinely even rounds ("contested": equal players alive *and* even economy) every model falls toward a coin flip. |
| **3** | `F3_heatmap.png` | Model × feature-set AUC (in-time, out-of-fold). |
| **4** | `F4_datagap.png` | Why firepower fails out-of-time. The team skill-sum, tightly clustered in training (mean 5.28, all five players known), collapses on the 2026 holdout (mean 3.66) because ≈30% of players have no database entry. The same feature means something different in 2026. |
| **5** | `holdout_2026.png` | In-time vs out-of-time AUC (left) — sets without firepower sit on the no-degradation line; EFB2 falls far below. Calibration on the holdout (right) under the 0.445 → 0.512 base-rate shift. |
| 6 | `mapcontrol_compare_*.png` | The three control representations on one round: Voronoi / grey (LOS+FOV+smoke) / territory (memory+decay). |
| 7 | `calibration.png` | Reliability diagram with bootstrap CIs. |
| 8 | `winprob_*.png` | A live win-probability curve with a 95% CI band. |

---

## ⚠️ TODO before submission
- [ ] Full citations (Xenopoulos/ESTA; pitch-control lineage; Choi et al. on esports calibration metrics; TCN/GAT refs).
- [ ] Re-run holdout after the 2026 skill scrape → report same-era **and** lagged-prior variants (§7.8).
- [ ] Decide whether to re-evaluate the deep models out-of-time on EB2 (no firepower).
- [ ] Compress to 9 pages LNCS (§7 will need tightening; move the extended metric battery to an appendix).
- [ ] Acknowledgement: PARCC allocation (required by terms of use).
- [ ] Author list / affiliations for Leu.
