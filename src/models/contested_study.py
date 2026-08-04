"""Study 2 - The contested-round ceiling.

Three analyses, all at the SNAPSHOT level (round-level contested is underpowered: a prior run gave a
+-0.05 AUC CI on ~200 even rounds). Match-level bootstrap throughout.

(A) Contested-AUC by alive-state and a graded-evenness sweep. How predictability changes as rounds
    become more even, broken out by 1v1 / 2v2 / 3v3 / 4v4 / 5v5.

(B) Information-saturation curve. Contested-AUC across increasingly rich representations
    (economy -> +spatial -> +bomb -> +firepower -> TCN -> Transformer). If it plateaus even for the
    models that see the most, the plateau is a ceiling, not a feature-engineering artifact.

(C) Matching-based Bayes-error estimate (the model-free ceiling). For each even snapshot we find its
    nearest neighbours in observable-state space FROM DIFFERENT MATCHES (same-match neighbours share
    the round label and are excluded), and measure (i) the near-twin outcome DISAGREEMENT rate and
    (ii) a k-NN ORACLE AUC = how well the local neighbour outcome-frequency predicts the true outcome.
    A predictor that perfectly memorised the local outcome distribution cannot beat the oracle AUC, so
    it is an empirical ceiling. Run across every even alive-state, plus an uneven control that should
    show LOW disagreement (validating the method detects determinism where it exists).

Snapshot selection is documented in outputs/bayes_error_matching.csv and printed at run time.
Usage: python src/models/contested_study.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import (  # noqa: E402
    FEATURE_SETS, ECONOMY_COLS, MAPCONTROL_COLS, TACTICAL, oof_predict)

TRAIN = ROOT / "data" / "training_dataset.parquet"
OOF_TCN = ROOT / "outputs" / "oof_tcn.parquet"
OOF_TF = ROOT / "outputs" / "oof_transformer.parquet"
STATE_COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL   # observable "situation" descriptor
SEED = 0
K = 25            # neighbours per query
K_BUF = 200       # over-fetch to allow same-match filtering
N_QUERY = 2500    # queries sampled per category
POOL_CAP = 25000  # neighbour-pool cap per category (subsample if larger), for runtime


def auc_ci(y, p, matches, B=500, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i, m in enumerate(matches):
        by.setdefault(m, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    pt = roc_auc_score(y, p) if (y.min() != y.max()) else np.nan
    boot = []
    for _ in range(B):
        s = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys = y[s]
        if ys.min() != ys.max():
            boot.append(roc_auc_score(ys, p[s]))
    lo, hi = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if boot else (np.nan, np.nan)
    return pt, lo, hi


def main():
    df = pl.read_parquet(TRAIN)
    y = df["ct_won"].to_numpy().astype(float)
    ca = df["ct_players_alive"].to_numpy(); ta = df["t_players_alive"].to_numpy()
    ce = df["ct_equipment_value"].to_numpy(); tee = df["t_equipment_value"].to_numpy()
    mid = df["match_id"].to_numpy()
    rnum = df["round_num"].to_numpy() if "round_num" in df.columns else np.zeros(len(y))
    even_alive = (ca == ta)
    even_econ = np.abs(ce - tee) <= 1500
    contested = even_alive & even_econ
    print(f"train {df.height} snaps / {df['match_id'].n_unique()} matches; "
          f"contested (equal alive & even econ) = {contested.mean()*100:.1f}%\n")

    # ---------- (A) contested-AUC by alive-state (generalist OOF) ----------
    print("=== (A) contested-AUC by alive-state (xgb EB2 OOF, generalist) ===")
    oof_eb2, _ = oof_predict(df, FEATURE_SETS["EB2"], "xgb")
    a_rows = []
    for k in [2, 4, 6, 8, 10]:
        lab = f"{k//2}v{k//2}"
        m = (ca == ta) & (ca + ta == k) & even_econ   # even alive + even econ at this headcount
        n = int(m.sum())
        if n < 200 or y[m].min() == y[m].max():
            print(f"  {lab:5s} n={n:6d}  (too few)"); continue
        pt, lo, hi = auc_ci(y[m], oof_eb2[m], mid[m])
        nr = len(set(zip(mid[m].tolist(), rnum[m].tolist())))
        a_rows.append({"state": lab, "n_snap": n, "n_rounds": nr,
                       "n_matches": len(set(mid[m].tolist())), "cAUC": pt, "lo": lo, "hi": hi,
                       "ct_winrate": float(y[m].mean())})
        print(f"  {lab:5s} n={n:6d} rounds={nr:5d}  cAUC {pt:.3f} ({lo:.3f},{hi:.3f})  "
              f"ctwin {y[m].mean():.3f}")
    pl.DataFrame(a_rows).write_csv(ROOT / "outputs" / "contested_by_alivestate.csv")

    # graded evenness: sweep the equipment-difference threshold (equal alive only)
    print("\n=== (A) graded evenness: contested-AUC vs equipment-diff threshold (equal alive) ===")
    g_rows = []
    for thr in [10000, 5000, 3000, 1500, 750, 250]:
        m = even_alive & (np.abs(ce - tee) <= thr)
        if m.sum() < 200:
            continue
        pt, lo, hi = auc_ci(y[m], oof_eb2[m], mid[m])
        g_rows.append({"equip_thr": thr, "n_snap": int(m.sum()), "cAUC": pt, "lo": lo, "hi": hi})
        print(f"  |dEquip|<= {thr:6d}  n={int(m.sum()):6d}  cAUC {pt:.3f} ({lo:.3f},{hi:.3f})")
    pl.DataFrame(g_rows).write_csv(ROOT / "outputs" / "contested_graded.csv")

    # ---------- (B) information-saturation curve on the contested set ----------
    print("\n=== (B) saturation: contested-AUC across representations (xgb classical + deep) ===")
    sat = []
    for st in ["A", "E", "EB2", "EFB2"]:
        oof, _ = oof_predict(df, FEATURE_SETS[st], "xgb")
        pt, lo, hi = auc_ci(y[contested], oof[contested], mid[contested])
        sat.append({"repr": st, "family": "classical", "cAUC": pt, "lo": lo, "hi": hi})
        print(f"  {st:6s} (xgb)      contested-AUC {pt:.3f} ({lo:.3f},{hi:.3f})")
    # deep OOF joined on (match_id, tick)
    key = df.select(["match_id", "tick"]).with_row_index("row")
    for name, f, pcol in [("TCN", OOF_TCN, "p_tcn"), ("Transformer", OOF_TF, "p_transformer")]:
        if not f.exists():
            continue
        d = pl.read_parquet(f)
        j = key.join(d.select(["match_id", "tick", pcol]), on=["match_id", "tick"], how="left").sort("row")
        p = j[pcol].to_numpy()
        ok = ~np.isnan(p)
        cm = contested & ok
        pt, lo, hi = auc_ci(y[cm], p[cm], mid[cm])
        sat.append({"repr": name, "family": "deep", "cAUC": pt, "lo": lo, "hi": hi})
        print(f"  {name:11s}(deep) contested-AUC {pt:.3f} ({lo:.3f},{hi:.3f})  [matched {ok.mean()*100:.0f}%]")
    pl.DataFrame(sat).write_csv(ROOT / "outputs" / "contested_saturation.csv")

    # ---------- (C) matching-based Bayes-error, across alive-states ----------
    print("\n=== (C) matching-based Bayes-error (near-twin disagreement + kNN oracle AUC) ===")
    print(f"    state descriptor = {len(STATE_COLS)} observable cols; K={K} neighbours from DIFFERENT "
          f"matches; up to {N_QUERY} queries / {POOL_CAP} pool per category\n")
    Xall = np.nan_to_num(df.select(STATE_COLS).to_numpy().astype(float))
    rng = np.random.default_rng(SEED)

    # categories: even alive-states (contested), plus an uneven control (man-advantage)
    cats = []
    for k in [2, 4, 6, 8, 10]:
        cats.append((f"{k//2}v{k//2} even", (ca == ta) & (ca + ta == k) & even_econ))
    cats.append(("all even (contested)", contested))
    cats.append(("CONTROL 1-man-adv", (np.abs(ca - ta) == 1) & (np.minimum(ca, ta) >= 1)))

    c_rows = []
    for label, mask in cats:
        idx = np.where(mask)[0]
        if len(idx) < 500:
            print(f"  {label:22s} n={len(idx):6d}  (too few, skipped)"); continue
        # neighbour pool (cap for runtime), standardised state space
        pool = idx if len(idx) <= POOL_CAP else rng.choice(idx, POOL_CAP, replace=False)
        Xp = StandardScaler().fit_transform(Xall[pool])
        pool_mid = mid[pool]; pool_y = y[pool]
        nn = NearestNeighbors(n_neighbors=min(K + K_BUF, len(pool))).fit(Xp)
        # queries (subsample)
        q = pool if len(pool) <= N_QUERY else rng.choice(pool, N_QUERY, replace=False)
        # map query rows into pool-standardised space
        Xq = StandardScaler().fit(Xall[pool]).transform(Xall[q])
        dist, nbr = nn.kneighbors(Xq)
        q_y = y[q]; q_mid = mid[q]
        local_p = np.full(len(q), np.nan); disagree = np.full(len(q), np.nan)
        for i in range(len(q)):
            cand = nbr[i]
            keep = [c for c in cand if pool_mid[c] != q_mid[i]][:K]   # exclude SAME-MATCH neighbours
            if len(keep) < 5:
                continue
            ny = pool_y[keep]
            local_p[i] = ny.mean()
            disagree[i] = np.mean(ny != q_y[i])
        good = ~np.isnan(local_p)
        D = float(np.nanmean(disagree))
        # kNN oracle AUC: does the local neighbour frequency predict the true outcome?
        oy, op, om = q_y[good], local_p[good], q_mid[good]
        oracle, olo, ohi = auc_ci(oy, op, om) if (oy.min() != oy.max()) else (np.nan, np.nan, np.nan)
        nr = len(set(zip(mid[idx].tolist(), rnum[idx].tolist())))
        c_rows.append({"category": label, "n_snap": int(len(idx)), "n_rounds": nr,
                       "n_matches": len(set(mid[idx].tolist())), "n_query": int(good.sum()),
                       "disagree_rate": D, "mean_local_p": float(np.nanmean(local_p)),
                       "oracle_AUC": oracle, "oracle_lo": olo, "oracle_hi": ohi,
                       "ct_winrate": float(y[idx].mean())})
        print(f"  {label:22s} n={len(idx):6d} rounds={nr:5d} matches={len(set(mid[idx].tolist())):4d} | "
              f"disagree {D:.3f} | oracle-AUC {oracle:.3f} ({olo:.3f},{ohi:.3f})")
    pl.DataFrame(c_rows).write_csv(ROOT / "outputs" / "bayes_error_matching.csv")
    print("\nwrote outputs/{contested_by_alivestate,contested_graded,contested_saturation,"
          "bayes_error_matching}.csv")


if __name__ == "__main__":
    main()
