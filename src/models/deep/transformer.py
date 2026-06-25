"""Causal Transformer encoder — per-second win-probability sequence model (architecture #8).

Same data/eval pipeline as the TCN (shares helpers from tcn.py): per-round sequence of the
aggregate per-second features -> a Transformer encoder with a CAUSAL self-attention mask, so
the prediction at second t attends only to seconds <= t (no within-round leakage, like the
causal TCN). Seq-to-seq: one P(CT win) per timestep. With end-padding + causal masking, valid
timesteps never attend to pad positions, so no key-padding mask is needed.

5-fold GroupKFold OOF + full metric suite + B=500 match-bootstrap CIs (via tcn.py helpers) and
optional --save-oof for the ensemble. Deps: torch, polars, numpy, sklearn.

Usage (Betty GPU): python src/models/deep/transformer.py --cv --epochs 40 --patience 6
"""
from __future__ import annotations
import argparse
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

import tcn  # same directory; reuse build_sequences / collect / metrics / bootstrap


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Transformer(nn.Module):
    def __init__(self, f, d=64, heads=4, layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(f, d)
        self.pe = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, activation="relu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                              # x [B,T,F] -> [B,T] logits
        h = self.pe(self.embed(x))
        T = x.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.enc(h, mask=mask)                     # causal self-attention
        return self.head(h).squeeze(-1)


def fit(Xs, M, Y, tri, vai, args, device, quiet=False):
    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.tensor(Xs[idx]), torch.tensor(M[idx]),
                                        torch.tensor(Y[idx])), batch_size=args.batch, shuffle=shuffle)
    tl, vl = loader(tri, True), loader(vai, False)
    model = Transformer(Xs.shape[2], args.dim, args.heads, args.layers, args.ff, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0; since = 0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, mb, yb in tl:
            opt.zero_grad()
            loss = tcn.masked_bce(model(xb.to(device)), yb.to(device), mb.to(device))
            loss.backward(); opt.step(); tot += loss.item() * len(yb)
        auc, _, _ = tcn.evaluate(model, vl, device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_dataset.parquet")
    ap.add_argument("--seq-len", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit-matches", type=int, default=0)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--save-oof", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    df = pl.read_parquet(args.data)
    if args.limit_matches:
        keep = df["match_id"].unique().to_list()[: args.limit_matches]
        df = df.filter(pl.col("match_id").is_in(keep))
    cols = [c for c in df.columns if c not in tcn.NON_FEATURES and c != "round_num"]
    X, M, Y, G, C, Tk = tcn.build_sequences(df, cols, args.seq_len)
    print(f"{len(Y)} rounds x {args.seq_len} steps x {len(cols)} features; {len(set(G))} matches; "
          f"dim {args.dim} heads {args.heads} layers {args.layers}\n")

    def standardize(tri):
        mt = M[tri].astype(bool)
        mu = X[tri][mt].mean(0); sd = X[tri][mt].std(0) + 1e-6
        return ((X - mu) / sd).astype(np.float32)

    if args.cv:
        aY, aP, aC, aG, aT = [], [], [], [], []
        for k, (tri, tei) in enumerate(GroupKFold(5).split(np.zeros(len(Y)), Y, G), 1):
            Xs = standardize(tri)
            g_tr = np.array(sorted(set(G[tri])))
            rng = np.random.default_rng(args.seed + k)
            inner = set(rng.choice(g_tr, max(1, int(len(g_tr) * 0.15)), replace=False))
            itr = np.array([i for i in tri if G[i] not in inner])
            iva = np.array([i for i in tri if G[i] in inner])
            print(f"fold {k}: train {len(itr)} / inner-val {len(iva)} / test {len(tei)}")
            model, _ = fit(Xs, M, Y, itr, iva, args, device, quiet=True)
            y, p, c, g, t = tcn.collect(model, Xs, M, Y, C, tei, device, args.batch, G=G, Tk=Tk)
            aY.append(y); aP.append(p); aC.append(c); aG.append(g); aT.append(t)
        Yf, Pf, Cf = np.concatenate(aY), np.concatenate(aP), np.concatenate(aC)
        Gf, Tf = np.concatenate(aG), np.concatenate(aT)
        print("\n" + tcn.metric_line("Transformer (OOF)", Yf, Pf, Cf))
        if args.bootstrap:
            tcn.print_bootstrap(tcn.block_bootstrap_metrics(Yf, Pf, Gf, Cf, args.bootstrap), args.bootstrap)
        print(tcn.BASELINE)
        if args.save_oof:
            pl.DataFrame({"match_id": Gf, "tick": Tf, "y": Yf, "p_transformer": Pf}).write_parquet(args.save_oof)
            print(f"saved OOF -> {args.save_oof}")
    else:
        rng = np.random.default_rng(args.seed)
        matches = np.array(sorted(set(G)))
        val_m = set(rng.choice(matches, max(1, int(len(matches) * args.val_frac)), replace=False))
        tri = np.array([i for i in range(len(G)) if G[i] not in val_m])
        vai = np.array([i for i in range(len(G)) if G[i] in val_m])
        Xs = standardize(tri)
        print(f"train {len(tri)} / val {len(vai)} rounds\n")
        model, best = fit(Xs, M, Y, tri, vai, args, device)
        y, p, c = tcn.collect(model, Xs, M, Y, C, vai, device, args.batch)
        print("\n" + tcn.metric_line("Transformer (val)", y, p, c))
        print(tcn.BASELINE)


if __name__ == "__main__":
    main()
