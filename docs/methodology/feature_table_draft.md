# Feature table (draft for paper Appendix) — every feature by pillar

Working inventory for the supplementary "complete feature list" table (task 8). 115 base features +
the new 4-column defuse-progress block (quarantined into EB2D/EFB2D). Source of truth: the `*_COLS`
constants in `src/features/`.

## Pillar 1 — Economy & combat state (17)
`time_elapsed_sec, ct_players_alive, t_players_alive, ct_health_total, t_health_total, ct_armor_total,
t_armor_total, ct_equipment_value, t_equipment_value, ct_economy_class, t_economy_class,
ct_defuse_kits, bomb_planted, ct_score, t_score, score_diff, round_num`

## Pillar 2 — Map control
- **Voronoi (9):** `ct_voronoi_control_pct, control_deficit, ct_a_site_control, ct_b_site_control,
  ct_banana_control, ct_mid_control, ct_ct_spawn_control, control_trend, control_volatility`
- **Grey / LOS+FOV+smoke (5):** `ct_los_control, t_los_control, contested_pct, grey_pct, ct_los_deficit`
- **Territory / memory-decay (5):** `ct_terr_control, t_terr_control, terr_contested_pct, terr_grey_pct,
  ct_terr_deficit`
- **Territory per-zone (5):** `terr_a_site_deficit, terr_b_site_deficit, terr_banana_deficit,
  terr_mid_deficit, terr_ct_spawn_deficit`

## Pillar 3 — Tactical readiness & bomb geometry
- **Tactical (28):** `ct_positional_entropy, t_positional_entropy, ct_a_site_players, ct_b_site_players,
  ct_banana_players, ct_mid_players, ct_ct_spawn_players, t_a_site_players, t_b_site_players,
  t_banana_players, t_mid_players, t_ct_spawn_players, ct_awp_alive, t_awp_alive, ct_awp_zone,
  t_awp_zone, bomb_carrier_zone, ct_smokes, ct_flashes, ct_fire, ct_he, ct_util_total, t_smokes,
  t_flashes, t_fire, t_he, t_util_total, utility_advantage`
- **Bomb plant geometry (6):** `bomb_site, bomb_plant_x, bomb_plant_y, min_ct_dist_to_bomb,
  min_ct_path_to_bomb, n_ct_near_bomb`
- **Bomb-live (8):** `bomb_state, bomb_dropped, ct_bomb_local_control, ct_bomb_local_deficit,
  min_ct_dist_to_bomb_live, min_t_dist_to_bomb_live, ct_closer_to_bomb, defuse_time_margin`
- **Defuse-race, counterfactual (4):** `defuse_margin_kit, n_ct_can_defuse, best_defuser_has_kit,
  defuse_contest_margin`
- **NEW — Defuse-progress, actual (4):** `defuse_in_progress, defuse_elapsed_sec, defuse_progress_frac,
  defuse_beats_fuse` — quarantined into EB2D/EFB2D only.

## Pillar 4 — Firepower / skill prior (20, v2 sums)
`ct_rating_sum, t_rating_sum, ct_adr_sum, t_adr_sum, ct_kast_mean, t_kast_mean, ct_hltv_firepower_sum,
t_hltv_firepower_sum, ct_entry_sum, t_entry_sum, ct_trading_sum, t_trading_sum, ct_opening_sum,
t_opening_sum, ct_clutch_score, t_clutch_score, ct_awp_sniping_skill, t_awp_sniping_skill,
ct_weighted_utility, t_weighted_utility`

## Interactions (4)
`ctrl_x_eveneco, terr_x_eveneco, ctrl_x_equalalive, terr_x_equalalive`

## Feature-set composition (which pillars each set uses)
- **A** = Economy
- **E** = A + Voronoi + Grey + Territory + Tactical (map control + tactical)
- **EB2** = E + Bomb(plant+live+defuse-race)   [the production set]
- **EFB2** = EB2 + Firepower
- **EB2D** = EB2 + Defuse-progress (NEW)
- **EFB2D** = EFB2 + Defuse-progress (NEW)

Presentation idea for the paper: one longtable grouped by pillar, columns = [Pillar, Feature, Short
description, In sets]. Keep the defuse-progress block visually distinct (it is quarantined and is a
curve-honesty feature, not a discrimination feature).
