# Standard model report

Data: 476595 snapshots, 220 matches, base P(CT win)=0.445. Bootstrap B=200.

## logreg — set E

- **Metrics:** AUC 0.8493 · log-loss 0.4605 · Brier 0.1564 · ECE 0.0157 · BSS 0.367 · contested-AUC 0.5895
- **Uncertainty:** AUC lift vs A = +0.0027 (95% CI +0.0014, +0.0042)  [significant]; per-round win-prob CI band via `winprob_chart.py --model logreg`.
- **Calibration:** reliability curve → `figures/reliability_logreg_E.png` (ECE 0.0157).
- **Interpretation (permutation importance, top 8):** min_ct_dist_to_bomb (tact +0.059), ct_equipment_value (econ +0.054), t_equipment_value (econ +0.047), ct_health_total (econ +0.042), t_players_alive (econ +0.032), t_health_total (econ +0.031), bomb_plant_y (tact +0.028), bomb_planted (econ +0.021)

## logreg — set ET

- **Metrics:** AUC 0.8493 · log-loss 0.4604 · Brier 0.1564 · ECE 0.0158 · BSS 0.367 · contested-AUC 0.5854
- **Uncertainty:** AUC lift vs A = +0.0027 (95% CI +0.0014, +0.0043)  [significant]; per-round win-prob CI band via `winprob_chart.py --model logreg`.
- **Calibration:** reliability curve → `figures/reliability_logreg_ET.png` (ECE 0.0158).
- **Interpretation (permutation importance, top 8):** min_ct_dist_to_bomb (tact +0.059), ct_equipment_value (econ +0.052), t_equipment_value (econ +0.046), ct_health_total (econ +0.043), t_players_alive (econ +0.031), t_health_total (econ +0.031), bomb_plant_y (tact +0.028), bomb_planted (econ +0.021)

## xgb — set E

- **Metrics:** AUC 0.8476 · log-loss 0.4618 · Brier 0.1569 · ECE 0.0121 · BSS 0.365 · contested-AUC 0.5854
- **Uncertainty:** AUC lift vs A = +0.0034 (95% CI +0.0021, +0.0048)  [significant]; per-round win-prob CI band via `winprob_chart.py --model xgb`.
- **Calibration:** reliability curve → `figures/reliability_xgb_E.png` (ECE 0.0121).
- **Interpretation (permutation importance, top 8):** t_equipment_value (econ +0.041), ct_equipment_value (econ +0.039), t_health_total (econ +0.035), ct_health_total (econ +0.024), min_ct_dist_to_bomb (tact +0.016), ct_armor_total (econ +0.013), t_armor_total (econ +0.011), t_players_alive (econ +0.007)

## xgb — set ET

- **Metrics:** AUC 0.8480 · log-loss 0.4613 · Brier 0.1567 · ECE 0.0130 · BSS 0.366 · contested-AUC 0.5851
- **Uncertainty:** AUC lift vs A = +0.0038 (95% CI +0.0026, +0.0052)  [significant]; per-round win-prob CI band via `winprob_chart.py --model xgb`.
- **Calibration:** reliability curve → `figures/reliability_xgb_ET.png` (ECE 0.0130).
- **Interpretation (permutation importance, top 8):** t_equipment_value (econ +0.040), ct_equipment_value (econ +0.039), t_health_total (econ +0.032), ct_health_total (econ +0.023), min_ct_dist_to_bomb (tact +0.015), t_armor_total (econ +0.012), ct_armor_total (econ +0.012), t_players_alive (econ +0.008)
