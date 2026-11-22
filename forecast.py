"""Future-position forecast with TabPFNRegressor quantiles (the uncertainty cone) and a calibration check."""
import math

import numpy as np

from sim import DT, ROOMS, Robot, teacher

HORIZONS = [1.0, 2.0, 3.0, 4.0]
QUANTILES = [0.1, 0.5, 0.9]


def rollout(room, steps=3000, noise=0.2, seed=0, start=None):
    rng = np.random.default_rng(seed)
    r = Robot(room, *(start or (None, None, None)))
    S, P, ep, e = [], [], [], 0
    for _ in range(steps):
        d = r.sense()
        S.append(d)
        P.append((r.x, r.y, r.heading))
        ep.append(e)
        a = teacher(d)
        r.step(rng.integers(5) if rng.random() < noise else a)
        if r.crashed:
            r, e = Robot(room), e + 1
    return np.array(S), np.array(P), np.array(ep)


def future_rows(S, P, ep, horizons=HORIZONS):
    X, Y = [], []
    for h in horizons:
        k = round(h / DT)
        for t in range(len(S) - k):
            if ep[t] != ep[t + k]:
                continue
            x, y, th = P[t]
            dx, dy = P[t + k][0] - x, P[t + k][1] - y
            c, s = math.cos(th), math.sin(th)
            X.append([*S[t], h])
            Y.append((c * dx + s * dy, -s * dx + c * dy))
    return np.array(X), np.array(Y)


class FuturePredictor:
    def __init__(self, max_context=1000, n_estimators=1, horizons=HORIZONS):
        self.max_context, self.n_estimators, self.horizons = max_context, n_estimators, horizons

    def fit(self, X, Y, seed=0):
        from tabpfn import TabPFNRegressor

        if len(X) > self.max_context:
            idx = np.random.default_rng(seed).choice(len(X), self.max_context, replace=False)
            X, Y = X[idx], Y[idx]
        self.models = [TabPFNRegressor(fit_mode="fit_with_cache", n_estimators=self.n_estimators)
                       .fit(X, Y[:, j]) for j in range(2)]
        return self

    def predict(self, sensors):
        Q = np.array([[*sensors, h] for h in self.horizons])
        out = []
        for m in self.models:
            q = m.predict(Q, output_type="quantiles", quantiles=QUANTILES)
            out.append(np.stack(q, axis=1))
        return out[0], out[1]


def to_world(robot_pose, fwd, lat):
    x, y, th = robot_pose
    c, s = math.cos(th), math.sin(th)
    return x + c * fwd - s * lat, y + s * fwd + c * lat


TRAIN_ROOMS = ["A"] + [f"R{500 + i}" for i in range(10)]


def train_default(rooms=TRAIN_ROOMS, steps=6600, noise=0.2):
    Xs, Ys = [], []
    for i, room in enumerate(rooms):
        S, P, ep = rollout(room, steps // len(rooms), noise, seed=i)
        X, Y = future_rows(S, P, ep)
        Xs.append(X)
        Ys.append(Y)
    return FuturePredictor().fit(np.concatenate(Xs), np.concatenate(Ys))


if __name__ == "__main__":
    import time

    t = time.time()
    fp = train_default()
    print(f"ogrenme (oda A + 10 rastgele oda): {time.time() - t:.1f} sn\n")

    for room in ["B", "C", "R1000"]:
        S, P, ep = rollout(room, 1200, noise=0.2, seed=1)
        cover = {h: [] for h in HORIZONS}
        width = {h: [] for h in HORIZONS}
        times = []
        kmax = round(HORIZONS[-1] / DT)
        for t in range(0, len(S) - kmax, 6):
            ts = time.time()
            fwd, lat = fp.predict(S[t])
            times.append(time.time() - ts)
            x, y, th = P[t]
            c, s = math.cos(th), math.sin(th)
            for i, h in enumerate(HORIZONS):
                k = round(h / DT)
                if ep[t] != ep[t + k]:
                    continue
                dx, dy = P[t + k][0] - x, P[t + k][1] - y
                f, l = c * dx + s * dy, -s * dx + c * dy
                cover[h].append((fwd[i, 0] <= f <= fwd[i, 2], lat[i, 0] <= l <= lat[i, 2]))
                width[h].append(lat[i, 2] - lat[i, 0])
        print(f"oda {room} (hic gorulmedi)  | tahmin {np.median(times) * 1000:.0f} ms")
        for h in HORIZONS:
            cv = np.array(cover[h])
            print(f"  {h:.1f} sn sonra: %10-%90 araligi gercegi kapsiyor -> ileri {cv[:, 0].mean():.0%}, "
                  f"yan {cv[:, 1].mean():.0%} (hedef ~%80) | ort. yan genislik {np.mean(width[h]):.2f} m")
