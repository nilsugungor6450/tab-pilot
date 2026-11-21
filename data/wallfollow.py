"""TableTop - Wall-Following robot (UCI) ile TabPFN deneyi.

Kullanim:
    python wallfollow.py sensor_readings_24.data          # 24 sensor
    python wallfollow.py sensor_readings_4.data           # 4 basit mesafe
    python wallfollow.py sensor_readings_4.data --lags 3 --ahead 5

--lags  : gecmis kac adimin okumalari ozellik olarak eklensin (kisa sureli hafiza)
--ahead : kac adim sonraki hareketi tahmin edelim (0 = simdiki hareket)
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--lags", type=int, default=0)
p.add_argument("--ahead", type=int, default=0)
p.add_argument("--max-context", type=int, default=3000)   # 8GB RAM icin guvenli sinir
p.add_argument("--no-tabpfn", action="store_true")
args = p.parse_args()

# 1. Veriyi oku. Basligi yok, son sutun sinif etiketi.
df = pd.read_csv(args.file, header=None)
sensors = df.columns[:-1]
df.columns = [f"s{i}" for i in range(len(sensors))] + ["label"]
feat_cols = [c for c in df.columns if c != "label"]

# 2. Gecmis okumalar (lag) ekle. Satirlar zaman sirali (9 olcum/saniye).
X = df[feat_cols].copy()
for k in range(1, args.lags + 1):
    lagged = df[feat_cols].shift(k)
    lagged.columns = [f"{c}_lag{k}" for c in feat_cols]
    X = pd.concat([X, lagged], axis=1)

# 3. Hedef: 'ahead' adim sonraki hareket.
y = df["label"].shift(-args.ahead)

ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok].to_numpy(), y[ok].to_numpy()

# 4. Zaman sirasina gore bol: ilk %75 egitim, son %25 test.
#    Rastgele bolme yapma: ardisik satirlar neredeyse ayni, sonuc sahte yuksek cikar.
cut = int(len(X) * 0.75)
X_tr, y_tr, X_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]

# TabPFN context'ini sinirla (esit aralikla ornekle, zaman dagilimi korunsun)
if len(X_tr) > args.max_context:
    idx = np.linspace(0, len(X_tr) - 1, args.max_context).astype(int)
    X_ctx, y_ctx = X_tr[idx], y_tr[idx]
else:
    X_ctx, y_ctx = X_tr, y_tr

print(f"dosya={args.file}  ozellik={X.shape[1]}  lags={args.lags}  ahead={args.ahead}")
print(f"egitim={len(X_tr)} (context={len(X_ctx)})  test={len(X_te)}\n")

models = {
    "En sik sinif": DummyClassifier(strategy="most_frequent"),
    "Lineer (LogReg)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}
if not args.no_tabpfn:
    from tabpfn import TabPFNClassifier
    models["TabPFN"] = TabPFNClassifier()

results = {}
for name, m in models.items():
    Xf, yf = (X_ctx, y_ctx) if name == "TabPFN" else (X_tr, y_tr)
    m.fit(Xf, yf)
    pred = m.predict(X_te)
    results[name] = pred
    print(f"{name:18s} accuracy={accuracy_score(y_te, pred):.3f}  "
          f"balanced={balanced_accuracy_score(y_te, pred):.3f}")

best = list(results)[-1]
print(f"\n--- {best} sinif bazinda ---")
print(classification_report(y_te, results[best], zero_division=0))