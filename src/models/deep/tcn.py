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
from sklearn.metrics import roc_auc_score

NON_FEATURES = {"match_id", "tick", "ct_won"}


# ----------------------------- data -----------------------------
def build_sequences(df: pl.DataFrame, cols, seq_len: int):
    """-> X[N,T,F] float32, M[N,T] mask, Y[N] label, G[N] match_id (for group split)."""
    X, M, Y, G = [], [], [], []
    for (mid, _rn), g in df.sort("tick").group_by(["match_id", "round_num"], maintain_order=True):
        feats = np.nan_to_num(g.select(cols).to_numpy().astype(np.float32))
        t = min(len(feats), seq_len)
        xx = np.zeros((seq_len, len(cols)), np.float32)
        mm = np.zeros(seq_len, np.float32)
        xx[:t] = feats[:t]; mm[:t] = 1.0
        X.append(xx); M.append(mm); Y.append(int(g["ct_won"][0])); G.append(mid)
    return np.stack(X), np.stack(M), np.array(Y, np.float32), np.array(G)


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
    X, M, Y, G = build_sequences(df, cols, args.seq_len)
    print(f"{len(Y)} rounds x {args.seq_len} steps x {len(cols)} features; "
          f"{df['match_id'].n_unique()} matches")

    # split by MATCH (no leakage); standardize on train real-timesteps
    rng = np.random.default_rng(args.seed)
    matches = np.array(sorted(set(G)))
    val_m = set(rng.choice(matches, size=max(1, int(len(matches) * args.val_frac)), replace=False))
    tr, va = np.array([g not in val_m for g in G]), np.array([g in val_m for g in G])
    mt = M[tr].astype(bool)
    mu = X[tr][mt].mean(0); sd = X[tr][mt].std(0) + 1e-6
    Xs = (X - mu) / sd

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.tensor(Xs[idx]), torch.tensor(M[idx]),
                                        torch.tensor(Y[idx])), batch_size=args.batch, shuffle=shuffle)
    tl, vl = loader(tr, True), loader(va, False)

    model = TCN(len(cols), args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TCN params: {n_params:,}; train {tr.sum()} / val {va.sum()} rounds; "
          f"dropout {args.dropout}, wd {args.weight_decay}, patience {args.patience}\n")

    ckpt = Path(args.checkpoint); ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0; since = 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, mb, yb in tl:
            opt.zero_grad()
            loss = masked_bce(model(xb.to(device)), yb.to(device), mb.to(device))
            loss.backward(); opt.step(); tot += loss.item() * len(yb)
        auc, _, _ = evaluate(model, vl, device)
        flag = ""
        if auc > best:
            best = auc; since = 0; flag = "  *best (saved)"
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": auc,
                        "cols": cols, "mu": mu, "sd": sd, "args": vars(args)}, ckpt)
        else:
            since += 1
        print(f"epoch {ep:3d}  train_loss {tot/tr.sum():.4f}  val_AUC {auc:.4f}  "
              f"({time.time()-t0:.1f}s){flag}")
        if since >= args.patience:
            print(f"early stop: no val-AUC improvement for {args.patience} epochs (best {best:.4f})")
            break
    print(f"\nbest val AUC {best:.4f} (saved {ckpt}); classical baseline logreg EFB2 = 0.8515")


if __name__ == "__main__":
    main()
