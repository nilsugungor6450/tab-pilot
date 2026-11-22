"""TableTop - tahmin ufku (horizon) grafigi.

Kullanim:
    python horizon.py sensor_readings_4.data
    python horizon.py sensor_readings_24.data --lags 3

Her ufuk icin (0 = simdi, 9 = 1 sn sonra, ...) TabPFN ve lineer modelin
balanced accuracy degerini hesaplar, horizon.png ve horizon.csv kaydeder.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--lags", type=int, default=3)
p.add_argument("--horizons", default="0,4,9,18,27,45")   # 9 adim = 1 saniye
p.add_argument("--max-context", type=int, default=3000)
p.add_argument("--batch", type=int, default=100)          # test seti parca boyutu
p.add_argument("--out", default="horizon")                 # cikti dosya adi
p.add_argument("--no-tabpfn", action="store_true")
args = p.parse_args()

df = pd.read_csv(args.file, header=None)
feat_cols = list(df.columns[:-1])
label = df.columns[-1]

X_all = df[feat_cols].copy()
for k in range(1, args.lags + 1):
    lagged = df[feat_cols].shift(k)
    lagged.columns = [f"{c}_lag{k}" for c in feat_cols]
    X_all = pd.concat([X_all, lagged], axis=1)


def split(ahead):
    y = df[label].shift(-ahead)
    ok = X_all.notna().all(axis=1) & y.notna()
    X, y = X_all[ok].to_numpy(), y[ok].to_numpy()
    cut = int(len(X) * 0.75)                 # zamana gore bolme
    return X[:cut], y[:cut], X[cut:], y[cut:]


rows = []
for h in [int(v) for v in args.horizons.split(",")]:
    X_tr, y_tr, X_te, y_te = split(h)
    lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    row = {"horizon_steps": h, "seconds": round(h / 9, 2),
           "linear": balanced_accuracy_score(y_te, lin.predict(X_te))}
    if not args.no_tabpfn:
        from tabpfn import TabPFNClassifier
        idx = np.linspace(0, len(X_tr) - 1, min(args.max_context, len(X_tr))).astype(int)
        m = TabPFNClassifier().fit(X_tr[idx], y_tr[idx])
        # Test setini parca parca tahmin et (8GB RAM'de MPS bellegi tasmasin)
        pred = np.concatenate([m.predict(X_te[i:i + args.batch])
                               for i in range(0, len(X_te), args.batch)])
        row["tabpfn"] = balanced_accuracy_score(y_te, pred)
        del m
        import gc, torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    rows.append(row)
    print(row)

res = pd.DataFrame(rows)
res.to_csv(f"{args.out}.csv", index=False)

plt.figure(figsize=(6, 4))
if "tabpfn" in res:
    plt.plot(res["seconds"], res["tabpfn"], "o-", label="TabPFN")
plt.plot(res["seconds"], res["linear"], "s--", label="Linear (LogReg)")
plt.axhline(0.25, color="gray", lw=1, ls=":", label="Chance level (4 classes)")
plt.xlabel("Prediction horizon (seconds ahead)")
plt.ylabel("Balanced accuracy")
plt.ylim(0, 1.02)
plt.title("How far can a table take a robot?")
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.out}.png", dpi=150)
print(f"\n{args.out}.png ve {args.out}.csv kaydedildi")
