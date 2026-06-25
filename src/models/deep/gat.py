"""GAT-style spatial deep model — masked multi-head self-attention over the player graph.

Consumes the RAW per-player trajectory state (x, y, velocity, facing, hp, armor, equip, kit,
side) for the (up to 10) alive players at each per-second snapshot, and predicts P(CT win).
Each player is a node; multi-head self-attention = a GAT on the fully-connected player graph
(no torch_geometric needed). Permutation-invariant (masked over dead/absent slots), so it learns
formations/relationships directly from positions — signal the aggregate-feature models can't use.

Per-snapshot (like the classical models): 5-fold GroupKFold OOF, full metric suite
(AUC / log-loss / Brier / ECE / BSS / contested-AUC), comparable to the matrix and the TCN.

Deps: torch, polars, numpy, scikit-learn (no torch_geometric). Reads data/trajectory_dataset.parquet.

Usage (on a Betty GPU node):
  python src/models/deep/gat.py --cv --epochs 40 --patience 6
  python src/models/deep/gat.py --limit-matches 20 --epochs 5      # smoke test
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold

MAXP = 10
PF = ["x", "y", "vx", "vy", "yaw", "hp", "armor", "equip", "kit", "isct"]
# After yaw->sin/cos the per-player vector is:
# [0]x [1]y [2]vx [3]vy [4]yaw_sin [5]yaw_cos [6]hp [7]armor [8]equip [9]kit [10]isct
NORM_IDX = [0, 1, 2, 3, 6, 7, 8]   # standardize continuous cols (x,y,vx,vy,hp,armor,equip)


# ----------------------------- metrics -----------------------------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def bss(y, p):
    base = y.mean(); ref = base * (1 - base)
    return float(1 - brier_score_loss(y, p) / ref) if ref > 0 else float("nan")


def metric_line(name, y, p, contested=None):
    s = (f"{name:16s} AUC {roc_auc_score(y, p):.4f}  logloss {log_loss(y, p):.4f}  "
         f"brier {brier_score_loss(y, p):.4f}  ECE {ece(y, p):.4f}  BSS {bss(y, p):.3f}")
    if contested is not None and contested.sum() > 50:
        s += f"  cAUC {roc_auc_score(y[contested], p[contested]):.4f}"
    return s


def block_bootstrap_auc(y, p, groups, B=500, seed=0):
    """Match-level block bootstrap CI on AUC over fixed OOF predictions (no retraining) —
    same procedure as the classical pipeline, so the CI is directly comparable."""
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
def load_data(path, limit_matches=0):
    df = pl.read_parquet(path)
    if limit_matches:
        keep = df["match_id"].unique().to_list()[:limit_matches]
        df = df.filter(pl.col("match_id").is_in(keep))
    n = df.height
    raw = df.select([f"p{i}_{f}" for i in range(MAXP) for f in PF]).to_numpy().astype(np.float32)
    raw = raw.reshape(n, MAXP, len(PF))                       # [N,10,10]
    mask = df.select([f"p{i}_alive" for i in range(MAXP)]).to_numpy().astype(bool)  # [N,10]
    yaw = np.radians(raw[:, :, 4])
    X = np.concatenate([raw[:, :, :4], np.sin(yaw)[..., None], np.cos(yaw)[..., None],
                        raw[:, :, 5:]], axis=2).astype(np.float32)  # [N,10,11] (yaw->sin,cos)
    y = df["ct_won"].to_numpy().astype(np.float32)
    g = df["match_id"].to_numpy()
    # contested = equal alive & even economy (computed from raw per-player equip/side)
    isct = raw[:, :, 9].astype(bool) & mask
    ist = (~raw[:, :, 9].astype(bool)) & mask
    eq = raw[:, :, 7]
    cont = (isct.sum(1) == ist.sum(1)) & (np.abs(np.where(isct, eq, 0).sum(1)
                                                 - np.where(ist, eq, 0).sum(1)) <= 1500)
    return X, mask, y, g, cont


# ----------------------------- model -----------------------------
class GATLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, h, kpm):
        a, _ = self.att(h, h, h, key_padding_mask=kpm, need_weights=False)
        h = self.n1(h + a)
        return self.n2(h + self.ffn(h))


class GAT(nn.Module):
    def __init__(self, f, d=64, heads=4, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(f, d)
        self.layers = nn.ModuleList([GATLayer(d, heads, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, mask):                       # x [B,10,F], mask [B,10] True=valid
        h = self.embed(x)
        kpm = ~mask                                   # key_padding_mask: True = ignore
        for lyr in self.layers:
            h = lyr(h, kpm)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # masked mean over valid players
        return self.head(pooled).squeeze(-1)          # [B] logit


# ----------------------------- train/eval -----------------------------
def fit(X, Mk, Y, tri, vai, args, device, quiet=False):
    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.tensor(X[idx]), torch.tensor(Mk[idx]),
                                        torch.tensor(Y[idx])), batch_size=args.batch, shuffle=shuffle)
    tl, vl = loader(tri, True), loader(vai, False)
    model = GAT(X.shape[2], args.dim, args.heads, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    best = 0.0; since = 0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, mb, yb in tl:
            opt.zero_grad()
            loss = bce(model(xb.to(device), mb.to(device)), yb.to(device))
            loss.backward(); opt.step(); tot += loss.item() * len(yb)
        auc = eval_auc(model, vl, device)
        flag = ""
        if auc > best:
            best = auc; since = 0; flag = "  *best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if not quiet:
            print(f"  epoch {ep:3d}  loss {tot/len(tl.dataset):.4f}  val_AUC {auc:.4f}  "
                  f"({time.time()-t0:.1f}s){flag}")
        if since >= args.patience:
            if not quiet:
                print(f"  early stop (best {best:.4f})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval(); ps, ys = [], []
    for xb, mb, yb in loader:
        ps.append(torch.sigmoid(model(xb.to(device), mb.to(device))).cpu().numpy()); ys.append(yb.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def predict(model, X, Mk, ridx, device, batch):
    model.eval(); ps = []
    for s in range(0, len(ridx), batch):
        b = ridx[s:s + batch]
        ps.append(torch.sigmoid(model(torch.tensor(X[b]).to(device),
                                      torch.tensor(Mk[b]).to(device))).cpu().numpy())
    return np.concatenate(ps)


def standardize(X, Mk, tri):
    flat = X[tri][Mk[tri]]                            # valid players in train
    mu = flat[:, NORM_IDX].mean(0); sd = flat[:, NORM_IDX].std(0) + 1e-6
    Xs = X.copy()
    Xs[:, :, NORM_IDX] = (Xs[:, :, NORM_IDX] - mu) / sd
    Xs *= Mk[..., None]                              # zero out dead/pad slots
    return Xs.astype(np.float32)


BASELINE = ("classical best (logreg EFB2): AUC 0.8515  ECE 0.016  cAUC 0.596   |   "
            "TCN (OOF): AUC 0.8493  ECE 0.009  cAUC 0.574")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectory_dataset.parquet")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit-matches", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="5-fold GroupKFold OOF (full metrics)")
    ap.add_argument("--bootstrap", type=int, default=500, help="match-level block bootstrap B for the OOF AUC CI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    X, Mk, Y, G, C = load_data(args.data, args.limit_matches)
    print(f"{len(Y)} snapshots x {MAXP} players x {X.shape[2]} per-player feats; "
          f"{len(set(G))} matches; dim {args.dim} heads {args.heads} layers {args.layers}\n")

    if args.cv:
        aY, aP, aC, aG = [], [], [], []
        for k, (tri, tei) in enumerate(GroupKFold(5).split(np.zeros(len(Y)), Y, G), 1):
            Xs = standardize(X, Mk, tri)
            g_tr = np.array(sorted(set(G[tri])))
            rng = np.random.default_rng(args.seed + k)
            inner = set(rng.choice(g_tr, max(1, int(len(g_tr) * 0.15)), replace=False))
            itr = np.array([i for i in tri if G[i] not in inner])
            iva = np.array([i for i in tri if G[i] in inner])
            print(f"fold {k}: train {len(itr)} / inner-val {len(iva)} / test {len(tei)}")
            model, _ = fit(Xs, Mk, Y, itr, iva, args, device, quiet=True)
            aP.append(predict(model, Xs, Mk, tei, device, args.batch))
            aY.append(Y[tei]); aC.append(C[tei]); aG.append(G[tei])
        Yf, Pf, Cf, Gf = (np.concatenate(aY), np.concatenate(aP),
                          np.concatenate(aC), np.concatenate(aG))
        print("\n" + metric_line("GAT (5-fold OOF)", Yf, Pf, Cf))
        if args.bootstrap:
            m, lo, hi = block_bootstrap_auc(Yf, Pf, Gf, args.bootstrap)
            print(f"  AUC match-level block bootstrap (B={args.bootstrap}): "
                  f"{m:.4f}  95% CI ({lo:.4f}, {hi:.4f})")
        print(BASELINE)
    else:
        rng = np.random.default_rng(args.seed)
        matches = np.array(sorted(set(G)))
        val_m = set(rng.choice(matches, max(1, int(len(matches) * args.val_frac)), replace=False))
        tri = np.array([i for i in range(len(G)) if G[i] not in val_m])
        vai = np.array([i for i in range(len(G)) if G[i] in val_m])
        Xs = standardize(X, Mk, tri)
        print(f"train {len(tri)} / val {len(vai)} snapshots\n")
        model, best = fit(Xs, Mk, Y, tri, vai, args, device)
        p = predict(model, Xs, Mk, vai, device, args.batch)
        print("\n" + metric_line("GAT (val)", Y[vai], p, C[vai]))
        print(BASELINE)


if __name__ == "__main__":
    main()
