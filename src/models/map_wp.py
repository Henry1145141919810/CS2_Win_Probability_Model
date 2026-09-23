"""Map win probability from round win probability: the state-model ladder (docs/studies/plan_round_to_map.md).

Team X = the side that is CT in the first half. Label = X wins the map (curated winner).
State at the start of round k: (a, b) = rounds won by X, Y (so k = a + b + 1), economy tiers
(e_X, e_Y) in {0 eco, 1 force, 2 partial, 3 full}; pistol rounds (1, 13) and every overtime round are
economy resets (pistol tiers / full buys), so the tiers only carry over in regulation non-pistol rounds.

Rungs (all 5-fold GroupKFold by match, everything estimated on the training folds only):
  M0  constant side rate, i.i.d. rounds            (the football-paper state model)
  M1  economy chain: p(side, pistol, e_X, e_Y) + empirical tier transitions T[outcome][tier->tier']
  M2  M1 + latent strength: logit shift from the Beta-posterior edge of X given (a, b), n0 by CV
  M3  residual check: XGBoost on [logit V_M2, score state, tiers, recent outcomes]
  M4  per-second chaining with the production round model (OOF logistic EB2)

Battery: round-start log-loss / Brier / AUC with paired match-block bootstrap, calibration by score
margin and by phase, structural checks, martingale increments + quadratic variation, extreme-path
benchmark (round-start and per-second paths), synthetic i.i.d. negative control, label-shift control,
learning curve. Out-of-time is NOT run here (touch-once; deferred until the in-time design is frozen).

Usage: python src/models/map_wp.py [--bootstrap 500] [--no-controls] [--no-persecond]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.t0_table import build as build_round_table  # noqa: E402
from models.pathwise_calibration import bench_tail_trough, pit, ks_upper_lower, ks_null_pvalue  # noqa: E402

OUT = ROOT / "outputs"
TIER_CUTS = [6000, 15000, 22000]        # eco | force | partial | full   (side equipment value)
NT = 4                                  # carry-over tiers
KMAX = 60                               # rounds beyond this are treated as a coin flip (never reached)
AMAX = 36
N0_GRID = [0, 5, 10, 20, 40, 80, 160]   # latent-strength prior strength (0 = M1)


# =============================================================================================
# format rules
# =============================================================================================
def x_is_ct(k: int) -> bool:
    """Side of X (first-half CT) at round k; verified on the data (side_schedule in the pilot)."""
    if k <= 12:
        return True
    if k <= 24:
        return False
    j = (k - 25) % 6
    period = (k - 25) // 6
    first_half_ct = (period % 2 == 1)
    return first_half_ct if j < 3 else (not first_half_ct)


def winner(a: int, b: int) -> int:
    """+1 if X has won the map at score (a, b), -1 if Y has, 0 otherwise.
    First to 13 in regulation (12-12 -> OT); in OT period n the target is 13 + 3n with the loser at
    most 11 + 3n (16-14, 19-17, ...)."""
    for hi, lo in ((a, b), (b, a)):
        if hi >= 13 and (hi - 13) % 3 == 0 and lo <= hi - 2:
            return 1 if hi == a else -1
    return 0


def is_reset(k: int) -> str | None:
    """'pistol' at rounds 1 and 13, 'ot' for every overtime round, else None (tiers carry over)."""
    if k in (1, 13):
        return "pistol"
    if k >= 25:
        return "ot"
    return None


def tier_of(equip: np.ndarray) -> np.ndarray:
    return np.digitize(equip, TIER_CUTS)


# =============================================================================================
# data
# =============================================================================================
SUPP = ROOT / "data" / "cs2_inferno_raw_supplement_v1"


def load_rounds(era: str = "train") -> pl.DataFrame:
    if era == "train":
        r, players, meta = build_round_table()
    else:   # the touch-once 2026 holdout: scored ONCE, disclosed (see --holdout)
        r, players, meta = build_round_table(
            era_dir=SUPP / "holdout_2026",
            per_second=SUPP / "holdout_2026" / "modelling" / "test_dataset_2026_per_second.parquet",
            demo_list=SUPP / "reference" / "demo_list_2026_test.csv")
    r = r.sort(["match_id", "round_num"])
    x = (r.filter(pl.col("round_num") <= 12).group_by("match_id")
          .agg(pl.col("ct_team_clan").mode().first().alias("x_clan")))
    r = r.join(x, on="match_id")
    r = r.with_columns(x_is_ct=(pl.col("ct_team_clan") == pl.col("x_clan")),
                       xw=(pl.col("round_winner_team_clan") == pl.col("x_clan")).cast(pl.Int64))
    r = r.with_columns(a=pl.col("xw").cum_sum().over("match_id") - pl.col("xw"),
                       b=(1 - pl.col("xw")).cum_sum().over("match_id") - (1 - pl.col("xw")))
    # curated map winner -> did X win
    def _norm(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())
    fin = r.group_by("match_id").agg(x_final=pl.col("xw").sum(), y_final=(1 - pl.col("xw")).sum())
    r = r.join(fin, on="match_id")

    def _x_won(s):
        cx, ca, cb, cw = _norm(s["x_clan"]), _norm(s["team_a"]), _norm(s["team_b"]), _norm(s["match_winner"])
        ha = cx == ca or cx in ca or ca in cx
        hb = cx == cb or cx in cb or cb in cx
        if ha != hb:
            picked = ca if ha else cb
            return int(picked == cw or picked in cw or cw in picked)
        return int(s["x_final"] > s["y_final"])
    r = r.with_columns(pl.struct(["x_clan", "team_a", "team_b", "match_winner", "x_final", "y_final"])
                         .map_elements(_x_won, return_dtype=pl.Int64).alias("x_won"))
    # economy tiers from X's / Y's perspective
    r = r.with_columns(
        x_equip=pl.when(pl.col("x_is_ct")).then(pl.col("ct_equipment_value")).otherwise(pl.col("t_equipment_value")),
        y_equip=pl.when(pl.col("x_is_ct")).then(pl.col("t_equipment_value")).otherwise(pl.col("ct_equipment_value")))
    r = r.with_columns(eX=pl.Series(tier_of(r["x_equip"].to_numpy())), eY=pl.Series(tier_of(r["y_equip"].to_numpy())),
                       k=pl.col("round_num"), pistol=pl.col("round_num").is_in([1, 13]).cast(pl.Int64),
                       ot=(pl.col("round_num") >= 25).cast(pl.Int64))
    r = r.with_columns(xw_prev=pl.col("xw").shift(1).over("match_id"))
    return r


# =============================================================================================
# per-round model + transitions (fitted on training folds only)
# =============================================================================================
def theta_hat(a, b, n0: float, mu: float = 0.5):
    return (n0 * mu + a) / (n0 + a + b) if n0 > 0 else np.full_like(np.asarray(a, dtype=float), 0.5)


def round_features(side_ct, pistol, ot, eX, eY, a, b, n0):
    """Design matrix for P(X wins round). Tiers enter as one-hot (ignored on pistol/OT rows because
    those rounds are resets: pistol -> tiers meaningless, OT -> always full)."""
    side_ct = np.asarray(side_ct, float); pistol = np.asarray(pistol, float); ot = np.asarray(ot, float)
    # overtime rounds are full buys on both sides with no carry-over: treat them as regulation
    # full-vs-full rounds (an OT-specific coefficient fitted on 290 rounds over-fits and leaks a
    # spurious X edge into every state near 12-12)
    eX = np.where(ot > 0, NT - 1, np.asarray(eX)); eY = np.where(ot > 0, NT - 1, np.asarray(eY))
    reg = (1 - pistol)
    cols = [side_ct, pistol]
    for t in range(NT):
        cols.append(reg * (eX == t)); cols.append(reg * (eY == t))
    cols.append(reg * (eX - eY))                      # tier gap, linear
    th = theta_hat(np.asarray(a, float), np.asarray(b, float), n0)
    cols.append(np.log(np.clip(th, 1e-6, 1 - 1e-6) / np.clip(1 - th, 1e-6, 1)))   # strength edge (0 when n0 = 0)
    return np.column_stack(cols)


class RoundModel:
    def __init__(self, n0: float):
        self.n0 = n0
        self.lr = LogisticRegression(C=1.0, max_iter=5000)

    def fit(self, df: pl.DataFrame):
        X = round_features(df["x_is_ct"], df["pistol"], df["ot"], df["eX"], df["eY"], df["a"], df["b"], self.n0)
        self.lr.fit(X, df["xw"].to_numpy())
        return self

    def prob(self, side_ct, pistol, ot, eX, eY, a, b):
        X = round_features(side_ct, pistol, ot, eX, eY, a, b, self.n0)
        return self.lr.predict_proba(X)[:, 1]


def fit_transitions(df: pl.DataFrame):
    """T[outcome][source][target] for regulation rounds whose NEXT round is a carry-over round.
    source 0..3 = tiers, 4 = pistol (rounds 1, 13). outcome 1 = the team won this round."""
    d = df.with_columns(nxt_eX=pl.col("eX").shift(-1).over("match_id"), nxt_eY=pl.col("eY").shift(-1).over("match_id"),
                        nxt_k=pl.col("k").shift(-1).over("match_id"))
    d = d.filter(pl.col("nxt_k").is_not_null() & (pl.col("nxt_k") == pl.col("k") + 1)
                 & (pl.col("nxt_k") <= 24) & ~pl.col("nxt_k").is_in([13]))
    T = np.ones((2, NT + 1, NT))     # Laplace smoothing
    for row in d.iter_rows(named=True):
        src_x = NT if row["pistol"] else row["eX"]; src_y = NT if row["pistol"] else row["eY"]
        T[row["xw"], src_x, row["nxt_eX"]] += 1
        T[1 - row["xw"], src_y, row["nxt_eY"]] += 1
    return T / T.sum(axis=2, keepdims=True)


# =============================================================================================
# dynamic program:  V[a, b, eX, eY] = P(X wins map | start of round a+b+1 with these tiers)
# =============================================================================================
class MapDP:
    def __init__(self, rm: RoundModel | None, T: np.ndarray | None, p_const: float | None = None):
        self.rm, self.T, self.p_const = rm, T, p_const
        self.V = np.full((AMAX + 2, AMAX + 2, NT, NT), np.nan)
        self._solve()

    def _p_round(self, a, b, k):
        """P(X wins round k) for every (eX, eY) tier pair -> array (NT, NT)."""
        side = x_is_ct(k); reset = is_reset(k)
        if self.p_const is not None:
            p = self.p_const if side else 1 - self.p_const
            return np.full((NT, NT), p)
        eX, eY = np.meshgrid(np.arange(NT), np.arange(NT), indexing="ij")
        p = self.rm.prob(np.full(NT * NT, side), np.full(NT * NT, reset == "pistol"), np.full(NT * NT, reset == "ot"),
                         eX.ravel(), eY.ravel(), np.full(NT * NT, a), np.full(NT * NT, b))
        return p.reshape(NT, NT)

    def _next_value(self, a, b, k, x_won: bool):
        """E over next tiers of V(a', b') at the start of round k+1, given the tiers at round k."""
        a2, b2 = (a + 1, b) if x_won else (a, b + 1)
        w = winner(a2, b2)
        if w:
            return np.full((NT, NT), 1.0 if w > 0 else 0.0)
        if k + 1 > KMAX:
            return np.full((NT, NT), 0.5)
        Vn = self.V[a2, b2]                       # (NT, NT) over next tiers
        reset = is_reset(k + 1)
        if reset == "pistol":
            return np.full((NT, NT), Vn[0, 0])    # pistol tiers: use tier 0 slot (the round model ignores tiers there)
        if reset == "ot":
            return np.full((NT, NT), Vn[NT - 1, NT - 1])
        if self.p_const is not None or self.T is None:
            return np.full((NT, NT), Vn.mean())   # M0: tiers are irrelevant
        # transitions from current tiers (rows) -> next tiers; pistol source when k is a pistol round
        src_is_pistol = is_reset(k) == "pistol"
        Tx = self.T[1 if x_won else 0]; Ty = self.T[0 if x_won else 1]
        if src_is_pistol:
            px = np.tile(Tx[NT], (NT, 1)); py = np.tile(Ty[NT], (NT, 1))
        else:
            px = Tx[:NT]; py = Ty[:NT]
        # E[V] = sum_{ex', ey'} px[ex, ex'] py[ey, ey'] Vn[ex', ey']
        return px @ Vn @ py.T

    def _solve(self):
        for k in range(KMAX, 0, -1):
            for a in range(0, min(k, AMAX + 1)):
                b = k - 1 - a
                if b < 0 or b > AMAX or winner(a, b):
                    continue
                p = self._p_round(a, b, k)
                self.V[a, b] = p * self._next_value(a, b, k, True) + (1 - p) * self._next_value(a, b, k, False)

    def value(self, a, b, eX, eY):
        a = np.asarray(a); b = np.asarray(b); eX = np.asarray(eX); eY = np.asarray(eY)
        out = np.empty(len(a))
        for i in range(len(a)):
            w = winner(int(a[i]), int(b[i]))
            if w:
                out[i] = 1.0 if w > 0 else 0.0
            elif a[i] > AMAX or b[i] > AMAX:
                out[i] = 0.5
            else:
                k = int(a[i] + b[i] + 1); reset = is_reset(k)
                ex, ey = (0, 0) if reset == "pistol" else ((NT - 1, NT - 1) if reset == "ot" else (int(eX[i]), int(eY[i])))
                out[i] = self.V[int(a[i]), int(b[i]), ex, ey]
        return out

    def chain(self, a, b, eX, eY, p_x_round):
        """Per-second chaining: P_map = p * E[V | X wins] + (1-p) * E[V | Y wins] with the tiers at round k."""
        out = np.empty(len(a))
        for i in range(len(a)):
            ai, bi, k = int(a[i]), int(b[i]), int(a[i] + b[i] + 1)
            reset = is_reset(k)
            ex, ey = (0, 0) if reset == "pistol" else ((NT - 1, NT - 1) if reset == "ot" else (int(eX[i]), int(eY[i])))
            vw = self._next_value(ai, bi, k, True)[ex, ey]; vl = self._next_value(ai, bi, k, False)[ex, ey]
            out[i] = p_x_round[i] * vw + (1 - p_x_round[i]) * vl
        return out


# =============================================================================================
# evaluation helpers
# =============================================================================================
def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                AUC=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan)


def cal_slope(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def paired_boot(groups, y, preds: dict, B=500, seed=0):
    rng = np.random.default_rng(seed); uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    ll = {n: [] for n in preds}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick]); yy = y[idx]
        for n, p in preds.items():
            ll[n].append(log_loss(yy, np.clip(p[idx], 1e-6, 1 - 1e-6), labels=[0, 1]))
    return {n: np.array(v) for n, v in ll.items()}


def q(a): return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def fit_fold_models(tr: pl.DataFrame, n0_grid):
    """Fit M0/M1/M2 ingredients on training rounds; choose n0 by inner CV on round-level log-loss."""
    p_const = float(tr.filter(pl.col("x_is_ct"))["xw"].mean() * 0.5 + (1 - tr.filter(~pl.col("x_is_ct"))["xw"].mean()) * 0.5)
    T = fit_transitions(tr)
    # inner CV for n0 (round-level, grouped by match)
    best_n0, best_ll = 0, np.inf
    g = tr["match_id"].to_numpy(); yv = tr["xw"].to_numpy()
    for n0 in n0_grid:
        oof = np.zeros(len(yv))
        for itr, ite in GroupKFold(4).split(np.zeros(len(yv)), yv, g):
            sub = tr[itr.tolist()]
            rm = RoundModel(n0).fit(sub)
            te = tr[ite.tolist()]
            oof[ite] = rm.prob(te["x_is_ct"], te["pistol"], te["ot"], te["eX"], te["eY"], te["a"], te["b"])
        llv = log_loss(yv, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1])
        if llv < best_ll - 1e-6:
            best_ll, best_n0 = llv, n0
    rm1 = RoundModel(0).fit(tr)
    rm2 = RoundModel(best_n0).fit(tr)
    return p_const, T, rm1, rm2, best_n0


def run_ladder(r: pl.DataFrame, n_splits=5, n0_grid=N0_GRID, verbose=True):
    """OOF map-WP at every round start for M0, M1, M2 (+ ingredients per fold for M3/M4)."""
    y = r["x_won"].to_numpy(); g = r["match_id"].to_numpy()
    P = {"M0": np.zeros(len(y)), "M1": np.zeros(len(y)), "M2": np.zeros(len(y)), "M2c": np.zeros(len(y))}
    per_fold = []
    for fold, (itr, ite) in enumerate(GroupKFold(n_splits).split(np.zeros(len(y)), y, g)):
        tr, te = r[itr.tolist()], r[ite.tolist()]
        p_const, T, rm1, rm2, n0 = fit_fold_models(tr, n0_grid)
        dp0 = MapDP(None, None, p_const=p_const); dp1 = MapDP(rm1, T); dp2 = MapDP(rm2, T)
        for name, dp in (("M0", dp0), ("M1", dp1), ("M2", dp2)):
            P[name][ite] = dp.value(te["a"], te["b"], te["eX"], te["eY"])
        # M2c: Platt recalibration of the chain logit, fitted on the training fold (the chain has
        # ~50 parameters, so its in-fold values are close to out-of-fold)
        ztr = dp2.value(tr["a"], tr["b"], tr["eX"], tr["eY"]); ztr = np.log(np.clip(ztr, 1e-6, 1 - 1e-6) / np.clip(1 - ztr, 1e-6, 1))
        cal = LogisticRegression(C=1e6, max_iter=5000).fit(ztr[:, None], tr["x_won"].to_numpy())
        zte = P["M2"][ite]; zte = np.log(np.clip(zte, 1e-6, 1 - 1e-6) / np.clip(1 - zte, 1e-6, 1))
        P["M2c"][ite] = cal.predict_proba(zte[:, None])[:, 1]
        per_fold.append(dict(fold=fold, test_idx=ite, p_const=p_const, n0=n0, dp0=dp0, dp1=dp1, dp2=dp2, T=T))
        if verbose:
            print(f"  fold {fold}: p_CT={p_const:.4f} n0={n0}  V(0-0)={dp2.V[0,0,0,0]:.3f}  "
                  f"V(12-12)={dp2.V[12,12,NT-1,NT-1]:.3f}", flush=True)
    return P, per_fold


def residual_gbm(r: pl.DataFrame, base: np.ndarray, n_splits=5):
    """M3: XGBoost on the chain logit + state, OOF by match."""
    from xgboost import XGBClassifier
    z = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
    X = np.column_stack([z, r["a"].to_numpy(), r["b"].to_numpy(), r["k"].to_numpy(), r["eX"].to_numpy(), r["eY"].to_numpy(),
                         r["x_is_ct"].to_numpy().astype(float), r["xw_prev"].fill_null(-1).to_numpy().astype(float),
                         r["x_equip"].to_numpy(), r["y_equip"].to_numpy()])
    y = r["x_won"].to_numpy(); g = r["match_id"].to_numpy(); oof = np.zeros(len(y))
    for itr, ite in GroupKFold(n_splits).split(X, y, g):
        m = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, min_child_weight=20, reg_lambda=10.0,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=0)
        m.fit(X[itr], y[itr]); oof[ite] = m.predict_proba(X[ite])[:, 1]
    return oof


def trajectory_checks(paths: list[np.ndarray], outcomes: np.ndarray, name: str):
    """Increment mean, quadratic-variation identity, and the extreme-path benchmark on map paths."""
    inc = np.concatenate([np.diff(p) for p in paths if len(p) > 1])
    p0 = np.array([p[0] for p in paths]); plast = np.array([p[-1] for p in paths])
    term = (outcomes - p0) ** 2
    qv = np.array([np.sum(np.diff(p) ** 2) for p in paths]) + (outcomes - plast) ** 2
    trough = np.array([np.min(p if o == 1 else 1 - p) for p, o in zip(paths, outcomes)])
    p0L = np.array([1 - p[0] if o == 1 else p[0] for p, o in zip(paths, outcomes)])
    rows = {}
    for yv in (0.05, 0.10, 0.20):
        obs = float((trough <= yv).mean()); ben = float(bench_tail_trough(p0L, yv).mean())
        rows[f"writeoff_le_{int(yv*100)}"] = obs; rows[f"bench_le_{int(yv*100)}"] = ben
    U = pit(1 - trough, p0L); du, dl = ks_upper_lower(U); pval = ks_null_pvalue(len(U), du, reps=500)
    out = dict(model=name, n_paths=len(paths), inc_mean=float(inc.mean()), inc_sd=float(inc.std()),
               qv_ratio=float(qv.mean() / term.mean()), term_over_implied=float(term.mean() / (p0 * (1 - p0)).mean()),
               D_upper=du, D_upper_p=pval, D_lower=dl, **rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-persecond", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="ONE disclosed out-of-time scoring on the 2026 maps")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    r = load_rounds()
    y = r["x_won"].to_numpy(); g = r["match_id"].to_numpy()
    print(f"rounds {r.height} maps {r['match_id'].n_unique()} X wins map {y.mean():.3f}  "
          f"tier shares {np.bincount(r['eX'].to_numpy(), minlength=NT)/r.height}")
    # ---- structural checks on the format rules --------------------------------------------------
    fin = r.group_by("match_id").agg(pl.col("x_final").first(), pl.col("y_final").first(), pl.col("x_won").first())
    ok = sum(1 for a, b, w in zip(fin["x_final"], fin["y_final"], fin["x_won"])
             if winner(int(a), int(b)) == (1 if w else -1))
    print(f"end rule reproduces the curated winner on {ok}/{fin.height} maps "
          f"(the rest lose their clinching round to validation trimming)")

    # ---- ladder --------------------------------------------------------------------------------
    print("\n== M0 / M1 / M2 (5-fold by match) ==")
    P, folds = run_ladder(r)
    P["M3"] = residual_gbm(r, P["M2"])
    n0s = [f["n0"] for f in folds]
    rows = []
    for name in ["M0", "M1", "M2", "M2c", "M3"]:
        m = metrics(y, P[name]); s, i = cal_slope(y, P[name])
        rows.append(dict(model=name, **m, cal_slope=s, cal_intercept=i))
        print(f"  {name}: logloss {m['logloss']:.4f} brier {m['brier']:.4f} AUC {m['AUC']:.4f} slope {s:.3f} intercept {i:+.3f}")
    print(f"  n0 chosen per fold: {n0s}")
    bs = paired_boot(g, y, P, B=args.bootstrap)
    for row in rows:
        n = row["model"]; row["ll_lo"], row["ll_hi"] = q(bs[n])
        for ref in ["M0", "M1", "M2"]:
            if ref != n:
                d = bs[n] - bs[ref]; row[f"dLL_vs_{ref}"] = float(d.mean()); row[f"dLL_vs_{ref}_lo"], row[f"dLL_vs_{ref}_hi"] = q(d)
    print("\n  paired dLL (95% CI):")
    for row in rows:
        s = "  ".join(f"vs {ref} {row[f'dLL_vs_{ref}']:+.4f} ({row[f'dLL_vs_{ref}_lo']:+.4f},{row[f'dLL_vs_{ref}_hi']:+.4f})"
                      for ref in ["M0", "M1", "M2"] if f"dLL_vs_{ref}" in row)
        print(f"  {row['model']}: {s}")
    pl.DataFrame(rows).write_csv(OUT / "map_wp_ladder.csv")

    # ---- calibration by margin and phase -----------------------------------------------------
    rr = r.with_columns(**{n: pl.Series(P[n]) for n in P}).with_columns(sd=(pl.col("a") - pl.col("b")).clip(-6, 6))
    cal = rr.group_by("sd").agg(n=pl.len(), realised=pl.col("x_won").mean(), **{n: pl.col(n).mean() for n in P}).sort("sd")
    print("\n== calibration by score margin (X - Y) ==");  print(cal.to_pandas().round(3).to_string(index=False))
    cal.write_csv(OUT / "map_wp_calibration_margin.csv")
    phases = [("round 1", 1, 1), ("half 1", 2, 12), ("round 13", 13, 13), ("half 2", 14, 24), ("overtime", 25, 99)]
    prow = []
    for name, lo, hi in phases:
        s = rr.filter((pl.col("k") >= lo) & (pl.col("k") <= hi)); yy = s["x_won"].to_numpy()
        prow.append(dict(phase=name, n=s.height, **{n: log_loss(yy, np.clip(s[n].to_numpy(), 1e-6, 1 - 1e-6), labels=[0, 1]) for n in P}))
    print("\n== log-loss by phase ==");  print(pl.DataFrame(prow).to_pandas().round(4).to_string(index=False))
    pl.DataFrame(prow).write_csv(OUT / "map_wp_phase.csv")
    # carry-over ablation by round index
    byk = rr.group_by("k").agg(n=pl.len(), **{f"ll_{n}": (-(pl.col("x_won") * pl.col(n).clip(1e-6, 1 - 1e-6).log()
                                                            + (1 - pl.col("x_won")) * (1 - pl.col(n).clip(1e-6, 1 - 1e-6)).log())).mean() for n in P}).sort("k")
    byk.write_csv(OUT / "map_wp_by_round.csv")

    # ---- structural checks on the fitted M2 -------------------------------------------------
    dp = folds[0]["dp2"]
    V = dp.V
    mono = all(np.all(np.diff(dp.value(np.arange(0, 14), np.full(14, bb), np.full(14, 3), np.full(14, 3))) >= -1e-9)
               for bb in range(12))
    print(f"\n== structural checks (fold 0, M2) ==\n  V(0-0) {V[0,0,0,0]:.3f}  V(12-12, full) {V[12,12,3,3]:.3f}  "
          f"V(12-0) {V[12,0,3,3]:.3f}  V(0-12) {V[0,12,3,3]:.3f}  monotone in a for b<12: {mono}")

    # ---- trajectory checks at round-start resolution -------------------------------------------
    traj = []
    for name in ["M0", "M2", "M3"]:
        paths, outs = [], []
        for mid, gdf in rr.group_by("match_id", maintain_order=True):
            paths.append(gdf[name].to_numpy()); outs.append(int(gdf["x_won"][0]))
        traj.append(trajectory_checks(paths, np.array(outs), f"{name} round-start"))
    print("\n== trajectory checks (round-start paths) ==")
    print(pl.DataFrame(traj).to_pandas().round(4).to_string(index=False))

    # ---- M4 per-second chaining ------------------------------------------------------------------
    if not args.no_persecond:
        oof = pl.read_parquet(OUT / "oof_noecon_logreg.parquet", columns=["match_id", "round_num", "tick", "p_EB2"])
        ts = pl.read_parquet(ROOT / "data" / "training_dataset.parquet", columns=["match_id", "round_num", "tick", "time_elapsed_sec"])
        oof = oof.join(ts, on=["match_id", "round_num", "tick"]).sort(["match_id", "round_num", "tick"])
        keyed = r.select(["match_id", "round_num", "a", "b", "eX", "eY", "x_is_ct", "x_won"])
        s = oof.join(keyed, on=["match_id", "round_num"], how="inner")
        s = s.with_columns(p_x=pl.when(pl.col("x_is_ct")).then(pl.col("p_EB2")).otherwise(1 - pl.col("p_EB2")))
        pmap = np.zeros(s.height)
        for f in folds:
            test_matches = set(r["match_id"][f["test_idx"].tolist()].unique().to_list())
            mask = s["match_id"].is_in(list(test_matches)).to_numpy()
            sub = s.filter(pl.Series(mask))
            pmap[mask] = f["dp2"].chain(sub["a"], sub["b"], sub["eX"], sub["eY"], sub["p_x"].to_numpy())
        s = s.with_columns(p_map=pl.Series(pmap))
        ys = s["x_won"].to_numpy()
        m = metrics(ys, pmap); sl, ic = cal_slope(ys, pmap)
        print(f"\n== M4 per-second chaining ==\n  snapshots {s.height}: logloss {m['logloss']:.4f} brier {m['brier']:.4f} "
              f"AUC {m['AUC']:.4f} slope {sl:.3f} intercept {ic:+.3f}")
        # consistency at freeze end: first snapshot of each round vs the round-start M2 value
        first = s.group_by(["match_id", "round_num"], maintain_order=True).agg(pl.col("p_map").first())
        j = rr.select(["match_id", "round_num", "M2"]).join(first, on=["match_id", "round_num"])
        print(f"  freeze-end consistency: mean |P_map(first snapshot) - M2| = {np.abs(j['p_map'].to_numpy() - j['M2'].to_numpy()).mean():.4f}")
        paths, outs = [], []
        for mid, gdf in s.group_by("match_id", maintain_order=True):
            paths.append(gdf["p_map"].to_numpy()); outs.append(int(gdf["x_won"][0]))
        t4 = trajectory_checks(paths, np.array(outs), "M4 per-second")
        traj.append(t4)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in t4.items() if k != "model"))
        s.select(["match_id", "round_num", "tick", "time_elapsed_sec", "x_won", "p_x", "p_map"]).write_parquet(OUT / "map_wp_persecond.parquet")
    pl.DataFrame(traj).write_csv(OUT / "map_wp_trajectory.csv")
    rr.select(["match_id", "round_num", "k", "a", "b", "eX", "eY", "x_is_ct", "x_won"] + list(P)).write_parquet(OUT / "map_wp_roundstart.parquet")

    # ---- controls ---------------------------------------------------------------------------------
    if not args.no_controls:
        print("\n== controls ==")
        # (1) label shift: pair every map's states with another map's winner
        rng = np.random.default_rng(0)
        maps = r["match_id"].unique().to_list(); perm = dict(zip(maps, rng.permutation(maps)))
        wmap = dict(zip(fin["match_id"], fin["x_won"]))
        y_shift = np.array([wmap[perm[m]] for m in g])
        print("  label-shift log-loss:", "  ".join(f"{n} {log_loss(y_shift, np.clip(P[n], 1e-6, 1-1e-6), labels=[0,1]):.4f}" for n in P), "(0.693 = no information)")
        # (2) synthetic i.i.d. maps from M0: refit M1/M2/M3 and compare with M0 on synthetic data
        synth = simulate_iid_maps(r, folds[0]["p_const"], n_maps=220, seed=1)
        Ps, _ = run_ladder(synth, verbose=False); Ps["M3"] = residual_gbm(synth, Ps["M2"])
        ysy = synth["x_won"].to_numpy(); gs = synth["match_id"].to_numpy()
        bss_ = paired_boot(gs, ysy, Ps, B=200)
        print("  synthetic i.i.d. control (paired dLL vs M0, should include 0):",
              "  ".join(f"{n} {float((bss_[n]-bss_['M0']).mean()):+.4f} ({q(bss_[n]-bss_['M0'])[0]:+.4f},{q(bss_[n]-bss_['M0'])[1]:+.4f})" for n in ["M1", "M2", "M3"]))
        # (3) learning curve
        lc = []
        for n_maps in [55, 110, 165, 220]:
            for seed in range(3 if n_maps < 220 else 1):
                pick = np.random.default_rng(seed).choice(maps, size=n_maps, replace=False)
                sub = r.filter(pl.col("match_id").is_in(pick.tolist()))
                Pl, _ = run_ladder(sub, verbose=False); Pl["M3"] = residual_gbm(sub, Pl["M2"])
                ysub = sub["x_won"].to_numpy()
                lc.append(dict(n_maps=n_maps, seed=seed, **{n: log_loss(ysub, np.clip(Pl[n], 1e-6, 1-1e-6), labels=[0,1]) for n in Pl}))
        lcd = pl.DataFrame(lc).group_by("n_maps").agg(**{n: pl.col(n).mean() for n in ["M0", "M1", "M2", "M2c", "M3"]}).sort("n_maps")
        print("  learning curve (OOF log-loss by number of maps):"); print(lcd.to_pandas().round(4).to_string(index=False))
        lcd.write_csv(OUT / "map_wp_learning_curve.csv")
    print("\nwrote outputs/map_wp_*.csv / .parquet")
    if args.holdout:
        run_holdout(r, P, args.bootstrap)


def run_holdout(r: pl.DataFrame, P_train: dict, bootstrap: int):
    """ONE disclosed out-of-time evaluation of the frozen in-time design on the 27 maps of 2026.
    Every ingredient is fitted on all 220 training maps; nothing is tuned here."""
    print("\n" + "=" * 90 + "\n== OUT-OF-TIME 2026 (touch-once; single disclosed evaluation) ==")
    te = load_rounds("holdout")
    yt = te["x_won"].to_numpy(); gt = te["match_id"].to_numpy()
    ct_rate = float(te.with_columns(ctw=pl.when(pl.col("x_is_ct")).then(pl.col("xw")).otherwise(1 - pl.col("xw")))["ctw"].mean())
    print(f"  2026: {te.height} round starts, {te['match_id'].n_unique()} maps, X wins map {yt.mean():.3f}, CT round rate {ct_rate:.3f}")
    p_const, T, rm1, rm2, n0 = fit_fold_models(r, N0_GRID)
    dp0 = MapDP(None, None, p_const=p_const); dp1 = MapDP(rm1, T); dp2 = MapDP(rm2, T)
    print(f"  fitted on all training maps: p_CT={p_const:.4f}, n0={n0}, V(0-0)={dp2.V[0,0,0,0]:.3f}")
    Pt = {"M0": dp0.value(te["a"], te["b"], te["eX"], te["eY"]),
          "M1": dp1.value(te["a"], te["b"], te["eX"], te["eY"]),
          "M2": dp2.value(te["a"], te["b"], te["eX"], te["eY"])}
    ztr = dp2.value(r["a"], r["b"], r["eX"], r["eY"]); ztr = np.log(np.clip(ztr, 1e-6, 1 - 1e-6) / np.clip(1 - ztr, 1e-6, 1))
    cal = LogisticRegression(C=1e6, max_iter=5000).fit(ztr[:, None], r["x_won"].to_numpy())
    z2 = np.log(np.clip(Pt["M2"], 1e-6, 1 - 1e-6) / np.clip(1 - Pt["M2"], 1e-6, 1))
    Pt["M2c"] = cal.predict_proba(z2[:, None])[:, 1]
    from xgboost import XGBClassifier

    def feats(df, base):
        z = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
        return np.column_stack([z, df["a"].to_numpy(), df["b"].to_numpy(), df["k"].to_numpy(), df["eX"].to_numpy(), df["eY"].to_numpy(),
                                df["x_is_ct"].to_numpy().astype(float), df["xw_prev"].fill_null(-1).to_numpy().astype(float),
                                df["x_equip"].to_numpy(), df["y_equip"].to_numpy()])
    m3 = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, min_child_weight=20, reg_lambda=10.0,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=0)
    m3.fit(feats(r, P_train["M2"]), r["x_won"].to_numpy()); Pt["M3"] = m3.predict_proba(feats(te, Pt["M2"]))[:, 1]
    # diagnostic only: the side constant re-estimated on 2026 (uses the test labels; labelled as such)
    Pt["M0 side@2026 (diagnostic)"] = MapDP(None, None, p_const=ct_rate).value(te["a"], te["b"], te["eX"], te["eY"])
    rows = []
    for n, p in Pt.items():
        m = metrics(yt, p); sl, ic = cal_slope(yt, p)
        rows.append(dict(model=n, **m, cal_slope=sl, cal_intercept=ic))
        print(f"  {n:28s} logloss {m['logloss']:.4f} brier {m['brier']:.4f} AUC {m['AUC']:.4f} slope {sl:.3f} intercept {ic:+.3f}")
    bs = paired_boot(gt, yt, Pt, B=bootstrap)
    for row in rows:
        n = row["model"]; row["ll_lo"], row["ll_hi"] = q(bs[n])
        if n != "M0":
            d = bs[n] - bs["M0"]; row["dLL_vs_M0"] = float(d.mean()); row["dLL_vs_M0_lo"], row["dLL_vs_M0_hi"] = q(d)
    print("  paired dLL vs M0 (95% CI, 27 maps):")
    for row in rows:
        if "dLL_vs_M0" in row:
            print(f"    {row['model']:28s} {row['dLL_vs_M0']:+.4f} ({row['dLL_vs_M0_lo']:+.4f},{row['dLL_vs_M0_hi']:+.4f})")
    tt = te.with_columns(**{n: pl.Series(p) for n, p in Pt.items()}).with_columns(sd=(pl.col("a") - pl.col("b")).clip(-4, 4))
    cal26 = tt.group_by("sd").agg(n=pl.len(), realised=pl.col("x_won").mean(), **{n: pl.col(n).mean() for n in ["M0", "M1", "M2c"]}).sort("sd")
    print("  calibration by margin (2026):"); print(cal26.to_pandas().round(3).to_string(index=False))
    traj = []
    for n in ["M0", "M1", "M2"]:
        paths, outs = [], []
        for mid, gdf in tt.group_by("match_id", maintain_order=True):
            paths.append(gdf[n].to_numpy()); outs.append(int(gdf["x_won"][0]))
        traj.append(trajectory_checks(paths, np.array(outs), f"{n} 2026 round-start"))
    print("  trajectory checks (27 paths; wide CIs):"); print(pl.DataFrame(traj).to_pandas().round(3).to_string(index=False))
    from models.train_pipeline import FEATURE_SETS, make_model
    trn = pl.read_parquet(ROOT / "data" / "training_dataset.parquet")
    tst = pl.read_parquet(ROOT / "data" / "test_dataset_2026.parquet")
    cols = FEATURE_SETS["EB2"]
    mdl = make_model("logreg").fit(np.nan_to_num(trn[cols].to_numpy().astype(float)), trn["ct_won"].to_numpy())
    p_eb2 = mdl.predict_proba(np.nan_to_num(tst[cols].to_numpy().astype(float)))[:, 1]
    s = (tst.select(["match_id", "round_num", "tick", "time_elapsed_sec"]).with_columns(p_EB2=pl.Series(p_eb2))
            .join(te.select(["match_id", "round_num", "a", "b", "eX", "eY", "x_is_ct", "x_won"]), on=["match_id", "round_num"], how="inner")
            .sort(["match_id", "round_num", "tick"]))
    s = s.with_columns(p_x=pl.when(pl.col("x_is_ct")).then(pl.col("p_EB2")).otherwise(1 - pl.col("p_EB2")))
    pmap = dp2.chain(s["a"], s["b"], s["eX"], s["eY"], s["p_x"].to_numpy())
    ys = s["x_won"].to_numpy(); m = metrics(ys, pmap); sl, ic = cal_slope(ys, pmap)
    print(f"  M4 per-second 2026: {s.height} snapshots logloss {m['logloss']:.4f} brier {m['brier']:.4f} AUC {m['AUC']:.4f} slope {sl:.3f} intercept {ic:+.3f}")
    paths, outs = [], []
    for mid, gdf in s.with_columns(p_map=pl.Series(pmap)).group_by("match_id", maintain_order=True):
        paths.append(gdf["p_map"].to_numpy()); outs.append(int(gdf["x_won"][0]))
    t4 = trajectory_checks(paths, np.array(outs), "M4 2026 per-second"); traj.append(t4)
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t4.items() if k != "model"))
    pl.DataFrame(rows).write_csv(OUT / "map_wp_holdout2026.csv"); cal26.write_csv(OUT / "map_wp_holdout2026_calibration.csv")
    pl.DataFrame(traj).write_csv(OUT / "map_wp_holdout2026_trajectory.csv")
    print("  wrote outputs/map_wp_holdout2026*.csv")


def simulate_iid_maps(r: pl.DataFrame, p_ct: float, n_maps=220, seed=0) -> pl.DataFrame:
    """Synthetic maps with i.i.d. rounds at the constant side rate, real side schedule and end rule,
    economy tiers drawn independently of outcomes (so no carry-over exists to be found)."""
    rng = np.random.default_rng(seed)
    tiers = r.filter(pl.col("pistol") == 0)["eX"].to_numpy()
    rows = []
    for m in range(n_maps):
        a = b = 0; k = 1; prev = None
        while not winner(a, b) and k <= KMAX:
            side = x_is_ct(k); p = p_ct if side else 1 - p_ct
            reset = is_reset(k)
            ex, ey = (0, 0) if reset == "pistol" else ((3, 3) if reset == "ot" else (int(rng.choice(tiers)), int(rng.choice(tiers))))
            xw = int(rng.random() < p)
            rows.append(dict(match_id=f"synth{m}", round_num=k, k=k, a=a, b=b, eX=ex, eY=ey, x_is_ct=side,
                             pistol=int(reset == "pistol"), ot=int(reset == "ot"), xw=xw, xw_prev=prev,
                             x_equip=float([3000, 10000, 18000, 26000][ex]), y_equip=float([3000, 10000, 18000, 26000][ey])))
            prev = xw; a += xw; b += 1 - xw; k += 1
        w = winner(a, b)
        for row in rows:
            if row["match_id"] == f"synth{m}":
                row["x_won"] = int(w > 0)
    return pl.DataFrame(rows)


if __name__ == "__main__":
    main()
