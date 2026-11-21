"""Ezber mi, genelleme mi? TabPFN'i hic gormedigi rastgele odalarda test et.

Kullanim:
    python generalize.py                                   # iki baglami karsilastir
    python generalize.py --contexts demos/teacher_A.csv --rooms 20

Test odalari R1000, R1001, ... (gosterim odalarindan tamamen ayri seed'ler).
Her odaya bir TabPFN robotu konur; tum robotlar her adimda TEK predict
cagrisiyla surulur. Ayni odalarda ogretmen de referans olarak surulur.

Olculenler:
  carpma         : robot duvara/sutuna carpti mi
  sag duvar bandi: adimlarin yuzde kacinda sag duvar 0.5-1.0 m arasinda (asil gorev)
Cikti: figures/generalize_<baglam>.png (oda izgarasi + robot izleri)
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policy import TabPFNPolicy
from sim import ROOMS, Robot, random_room, teacher

p = argparse.ArgumentParser()
p.add_argument("--contexts", default="demos/teacher_A.csv,demos/teacher_mixed.csv")
p.add_argument("--rooms", type=int, default=30)
p.add_argument("--steps", type=int, default=600)
args = p.parse_args()

rooms = [random_room(1000 + i) for i in range(args.rooms)]
RIGHT = 18   # 18 * 15 = 270 derece = robotun sagi


def run(drive):
    """drive(S) -> eylemler. Tum odalarda paralel surus."""
    robots = [Robot(r) for r in rooms]
    right = [[] for _ in robots]
    for _ in range(args.steps):
        alive = [i for i, r in enumerate(robots) if not r.crashed]
        if not alive:
            break
        S = np.array([robots[i].sense() for i in alive])
        for i, s, a in zip(alive, S, drive(S)):
            right[i].append(s[RIGHT])
            robots[i].step(int(a))
    band = [np.mean([(0.5 < d < 1.0) for d in rr[50:]]) if len(rr) > 50 else 0 for rr in right]
    return robots, band


def report(name, robots, band):
    crashed = sum(r.crashed for r in robots)
    print(f"  {name:28s} carpma {crashed:2d}/{len(robots)} | sag duvar bandinda {np.mean(band):.0%}")


def draw(robots, title, path):
    n = len(robots)
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 2.6, np.ceil(n / cols) * 2.6))
    for ax, r in zip(axes.flat, robots):
        for a, b in ROOMS[r.room]["walls"]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#1b1f3a", lw=1.6)
        tr = np.array(r.trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#d64545" if r.crashed else "#2e9e6a", lw=1)
        ax.plot(*tr[0], "o", color="#5b6cff", ms=4)
        if r.crashed:
            ax.plot(*tr[-1], "x", color="#d64545", ms=9, mew=2.5)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


print(f"{len(rooms)} hic gorulmemis rastgele oda, {args.steps} adim\n")
robots, band = run(lambda S: [teacher(s) for s in S])
report("ogretmen (referans)", robots, band)

for ctx in args.contexts.split(","):
    t = time.time()
    pol = TabPFNPolicy().fit_csv(ctx)
    robots, band = run(lambda S: pol.act(S)[0])
    label = os.path.basename(ctx).removesuffix(".csv")
    report(f"TabPFN, baglam={label}", robots, band)
    crashed = sum(r.crashed for r in robots)
    out = f"figures/generalize_{label}.png"
    draw(robots, f"TabPFN in {len(robots)} unseen rooms (context: {label}, {pol.n_context} rows) "
                 f"- {len(robots) - crashed}/{len(robots)} without a crash", out)
    print(f"  {'':28s} {time.time() - t:.0f} sn, sekil -> {out}")
