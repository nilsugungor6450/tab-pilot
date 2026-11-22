"""Record teacher demonstrations as a CSV table.

Usage:
    python record.py --room A,R500,R501 --steps 600 --noise 0.2
"""
import argparse
import os

import numpy as np
import pandas as pd

from sim import ACTION_NAMES, N_SENSORS, Robot, teacher


def record(room="A", steps=3000, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    r = Robot(room)
    rows = []
    for _ in range(steps):
        d = r.sense()
        a = teacher(d)
        rows.append([*d, ACTION_NAMES[a]])
        act = rng.integers(len(ACTION_NAMES)) if rng.random() < noise else a
        r.step(act)
        if r.crashed:
            r = Robot(room)
    cols = [f"s{i}" for i in range(N_SENSORS)] + ["action"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room", default="A")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rooms = args.room.split(",")
    df = pd.concat([record(r, args.steps, args.noise, seed=i) for i, r in enumerate(rooms)])
    out = args.out or f"demos/teacher_{args.room.replace(',', '_')}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} satir -> {out}")
    print(df["action"].value_counts().to_string())
