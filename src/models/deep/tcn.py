"""Causal TCN — per-second round win-probability sequence model (Betty GPU test).

Treats each round as a sequence of per-second snapshots and predicts P(CT win) at EVERY
second with a Temporal Convolutional Network. Convolutions are CAUSAL (left-padded then
cropped), so the prediction at second t uses only seconds <= t — no within-round future
leakage, matching a live win-probability use case.

Design (from CLAUDE_CONTEXT.md): 3 residual temporal blocks, dilations 1/2/4, kernel 3,
hidden 64, dropout 0.2. Sequence-to-sequence: input [B,T,F] -> logit per timestep [B,T];
masked BCE over real (non-pad) timesteps. Split by match (no leakage). Reports OOF-style
val AUC over valid timesteps vs the classical baseline (logreg EB2 ~0.851).

Dependencies: torch, polars, numpy, scikit-learn ONLY (no awpy/scipy) so it is light on the
cluster. Features = every column except match_id / tick / ct_won.

Usage (on a Betty GPU node via Slurm — never the login node):
  python src/models/deep/tcn.py --data data/training_dataset.parquet \
      --epochs 30 --checkpoint checkpoints/tcn.pt
  # quick smoke test on a subset:
  python src/models/deep/tcn.py --limit-matches 20 --epochs 3
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold

NON_FEATURES = {"match_id", "tick", "ct_won"}


# ----------------------------- metrics (self-contained, light deps) -----------------------------
def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def bss(y, p) -> float:
    base = y.mean(); ref = base * (1 - base)
    return float(1 - brier_score_loss(y, p) / ref) if ref > 0 else float("nan")


def metric_line(name, y, p, contested=None) -> str:
    s = (f"{name:16s} AUC {roc_auc_score(y, p):.4f}  logloss {log_loss(y, p):.4f}  "
         f"brier {brier_score_loss(y, p):.4f}  ECE {ece(y, p):.4f}  BSS {bss(y, p):.3f}")
    if contested is not None and contested.sum() > 50:
        s += f"  cAUC {roc_auc_score(y[contested], p[contested]):.4f}"
    return s


def block_bootstrap_auc(y, p, groups, B=500, seed=0):
    """Match-level block bootstrap CI on AUC over fixed OOF predictions (same as classical)."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    aucs = []
    for _ in range(B):
        samp = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys = y[samp]
        if ys.min() != ys.max():
            aucs.append(roc_auc_score(ys, p[samp]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)


# ----------------------------- data -----------------------------
def build_sequences(df: pl.DataFrame, cols, seq_len: int):
    """-> X[N,T,F], M[N,T] mask, Y[N] label, G[N] match_id, C[N,T] contested-flag (per timestep)."""
    X, M, Y, G, C = [], [], [], [], []
    for (mid, _rn), g in df.sort("tick").group_by(["match_id", "round_num"], maintain_order=True):
        feats = np.nan_to_num(g.select(cols).to_numpy().astype(np.float32))
        t = min(len(feats), seq_len)
        xx = np.zeros((seq_len, len(cols)), np.float32)
        mm = np.zeros(seq_len, np.float32)
        xx[:t] = feats[:t]; mm[:t] = 1.0
        ca = g["ct_players_alive"].to_numpy(); ta = g["t_players_alive"].to_numpy()
        ce = g["ct_equipment_value"].to_numpy(); te = g["t_equipment_value"].to_numpy()
        cont = (ca == ta) & (np.abs(ce - te) <= 1500)         # contested: equal alive & even econ
        cc = np.zeros(seq_len, bool); cc[:t] = cont[:t]
        X.append(xx); M.append(mm); Y.append(int(g["ct_won"][0])); G.append(mid); C.append(cc)
    return (np.stack(X), np.stack(M), np.array(Y, np.float32), np.array(G), np.stack(C))


# ----------------------------- model -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(cin, cout, k, padding=self.pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(cout, cout, k, padding=self.pad, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def _causal(self, c, x):                 # crop the right (future) side -> causal
        return c(x)[:, :, : -self.pad] if self.pad else c(x)

    def forward(self, x):
        out = self.drop(torch.relu(self._causal(self.conv1, x)))
        out = self.drop(torch.relu(self._causal(self.conv2, out)))
        res = x if self.down is None else self.down(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_features, hidden=64, levels=(1, 2, 4), k=3, dropout=0.2):
        super().__init__()
        chans = [n_features] + [hidden] * len(levels)
        self.blocks = nn.ModuleList(
            TemporalBlock(chans[i], chans[i + 1], k, d, dropout) for i, d in enumerate(levels))
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):                      # x [B,T,F]
        h = x.transpose(1, 2)                  # -> [B,F,T]
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)         # [B,T] logits


# ----------------------------- train/eval -----------------------------
def masked_bce(logits, y, m):
    y = y[:, None].expand_as(logits)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * m).sum() / m.sum()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps, ys = [], []
    for xb, mb, yb in loader:
        logit = model(xb.to(device))
        p = torch.sigmoid(logit).cpu().numpy()
        mm = mb.numpy().astype(bool)
        yb = yb.numpy()[:, None].repeat(p.shape[1], 1)
        ps.append(p[mm]); ys.append(yb[mm])
    p, y = np.concatenate(ps), np.concatenate(ys)
    return roc_auc_score(y, p), p, y


def fit(Xs, M, Y, tri, vai, args, device, ckpt=None, cols=None, quiet=False):
    """Train TCN with early stopping on the vai index; return (best_model, best_val_auc)."""
    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.tensor(Xs[idx]), torch.tensor(M[idx]),
                                        torch.tensor(Y[idx])), batch_size=args.batch, shuffle=shuffle)
    tl, vl = loader(tri, True), loader(vai, False)
    n_tr = len(tl.dataset)
    model = TCN(Xs.shape[2], args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0; since = 0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, mb, yb in tl:
            opt.zero_grad()
            loss = masked_bce(model(xb.to(device)), yb.to(device), mb.to(device))
            loss.backward(); opt.step(); tot += loss.item() * len(yb)
        auc, _, _ = evaluate(model, vl, device)
        flag = ""
        if auc > best:
            best = auc; since = 0; flag = "  *best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt:
                torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": auc,
                            "cols": cols, "args": vars(args)}, ckpt)
        else:
            since += 1
        if not quiet:
            print(f"  epoch {ep:3d}  train_loss {tot/n_tr:.4f}  val_AUC {auc:.4f}  "
                  f"({time.time()-t0:.1f}s){flag}")
        if since >= args.patience:
            if not quiet:
                print(f"  early stop (best {best:.4f})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def collect(model, Xs, M, Y, C, ridx, device, batch, G=None):
    """Flattened (y, p, contested[, match_id]) over valid (non-pad) timesteps for round indices."""
    model.eval(); ys, ps, cs, gs = [], [], [], []
    for s in range(0, len(ridx), batch):
        b = ridx[s:s + batch]
        p = torch.sigmoid(model(torch.tensor(Xs[b]).to(device))).cpu().numpy()
        mm = M[b].astype(bool)
        yb = np.repeat(Y[b][:, None], p.shape[1], axis=1)
        ps.append(p[mm]); ys.append(yb[mm]); cs.append(C[b][mm])
        if G is not None:
            gs.append(np.repeat(G[b][:, None], p.shape[1], axis=1)[mm])
    out = (np.concatenate(ys), np.concatenate(ps), np.concatenate(cs).astype(bool))
    return out + (np.concatenate(gs),) if G is not None else out


BASELINE = "classical best (logreg EFB2): AUC 0.8515  logloss 0.4559  brier 0.1552  ECE 0.016  cAUC 0.596"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_dataset.parquet")
    ap.add_argument("--seq-len", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.3)        # was 0.2 -> fight overfitting
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)  # L2 regularization
    ap.add_argument("--patience", type=int, default=6,           # early stopping
                    help="stop if val AUC doesn't improve for this many epochs")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit-matches", type=int, default=0, help="subset N matches for a smoke test")
    ap.add_argument("--checkpoint", default="checkpoints/tcn.pt")
    ap.add_argument("--cv", action="store_true",
                    help="5-fold GroupKFold OOF (full metrics over ALL rounds; comparable to classical)")
    ap.add_argument("--bootstrap", type=int, default=500, help="match-level block bootstrap B for OOF AUC CI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    df = pl.read_parquet(args.data)
    if args.limit_matches:
        keep = df["match_id"].unique().to_list()[: args.limit_matches]
        df = df.filter(pl.col("match_id").is_in(keep))
    cols = [c for c in df.columns if c not in NON_FEATURES and c != "round_num"]
    X, M, Y, G, C = build_sequences(df, cols, args.seq_len)
    print(f"{len(Y)} rounds x {args.seq_len} steps x {len(cols)} features; "
          f"{df['match_id'].n_unique()} matches; dropout {args.dropout} wd {args.weight_decay}\n")

    def standardize(train_idx):
        mt = M[train_idx].astype(bool)
        mu = X[train_idx][mt].mean(0); sd = X[train_idx][mt].std(0) + 1e-6
        return ((X - mu) / sd).astype(np.float32)

    if args.cv:
        # 5-fold GroupKFold by match -> out-of-fold predictions over EVERY round (no leakage),
        # so the metric suite is directly comparable to the classical matrix.
        aY, aP, aC, aG = [], [], [], []
        for k, (tri, tei) in enumerate(GroupKFold(5).split(np.zeros(len(Y)), Y, G), 1):
            Xs = standardize(tri)
            g_tr = np.array(sorted(set(G[tri])))           # carve an inner val for early stopping
            rng = np.random.default_rng(args.seed + k)
            inner = set(rng.choice(g_tr, max(1, int(len(g_tr) * 0.15)), replace=False))
            itr = np.array([i for i in tri if G[i] not in inner])
            iva = np.array([i for i in tri if G[i] in inner])
            print(f"fold {k}: train {len(itr)} / inner-val {len(iva)} / test {len(tei)}")
            model, _ = fit(Xs, M, Y, itr, iva, args, device, ckpt=None, cols=cols, quiet=True)
            y, p, c, g = collect(model, Xs, M, Y, C, tei, device, args.batch, G=G)
            aY.append(y); aP.append(p); aC.append(c); aG.append(g)
        Yf, Pf, Cf = np.concatenate(aY), np.concatenate(aP), np.concatenate(aC)
        Gf = np.concatenate(aG)
        print("\n" + metric_line("TCN (5-fold OOF)", Yf, Pf, Cf))
        if args.bootstrap:
            m, lo, hi = block_bootstrap_auc(Yf, Pf, Gf, args.bootstrap)
            print(f"  AUC match-level block bootstrap (B={args.bootstrap}): "
                  f"{m:.4f}  95% CI ({lo:.4f}, {hi:.4f})")
        print(BASELINE)
    else:
        # single match-level split (fast iteration)
        rng = np.random.default_rng(args.seed)
        matches = np.array(sorted(set(G)))
        val_m = set(rng.choice(matches, max(1, int(len(matches) * args.val_frac)), replace=False))
        tri = np.array([i for i in range(len(G)) if G[i] not in val_m])
        vai = np.array([i for i in range(len(G)) if G[i] in val_m])
        Xs = standardize(tri)
        ckpt = Path(args.checkpoint); ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"train {len(tri)} / val {len(vai)} rounds\n")
        model, best = fit(Xs, M, Y, tri, vai, args, device, ckpt=ckpt, cols=cols)
        y, p, c = collect(model, Xs, M, Y, C, vai, device, args.batch)
        print("\n" + metric_line("TCN (val)", y, p, c))
        print(BASELINE)
        print(f"(best checkpoint saved to {ckpt})")


if __name__ == "__main__":
    main()
