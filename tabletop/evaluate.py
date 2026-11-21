"""Ekransiz test: TabPFN robotu hic gormedigi odalarda ne kadar iyi suruyor?

Kullanim:
    python evaluate.py                         # demos/teacher_A.csv ile B ve C odalari
    python evaluate.py --rooms B --steps 600 --robots 8

Ayni anda --robots kadar robotu farkli baslangiclardan surer (hepsi TEK
predict cagrisinda, yani neredeyse bedava). Her robot icin carpma olup
olmadigi ve ogretmenle ayni karari verme orani raporlanir.
"""
import argparse
import math
import time

import numpy as np

from policy import TabPFNPolicy
from sim import ROOMS, Robot, teacher

p = argparse.ArgumentParser()
p.add_argument("--demos", default="demos/teacher_A.csv")
p.add_argument("--rooms", default="B,C")
p.add_argument("--steps", type=int, default=500)
p.add_argument("--robots", type=int, default=6)
args = p.parse_args()

t0 = time.time()
policy = TabPFNPolicy().fit_csv(args.demos)
print(f"baglam: {policy.n_context} satir, yukleme {time.time() - t0:.1f} sn\n")

for room in args.rooms.split(","):
    sx, sy, sh = ROOMS[room]["start"]
    # Robotlari ogretmenin rotasi uzerindeki farkli noktalardan baslat
    guide, starts = Robot(room), []
    for t in range(args.robots * 150):
        if t % 150 == 0:
            starts.append((guide.x, guide.y, guide.heading))
        guide.step(teacher(guide.sense()))
    robots = [Robot(room, x, y, h) for x, y, h in starts]

    agree, total, step_times = 0, 0, []
    for t in range(args.steps):
        alive = [r for r in robots if not r.crashed]
        if not alive:
            break
        S = np.array([r.sense() for r in alive])
        ts = time.time()
        acts, _ = policy.act(S)
        step_times.append(time.time() - ts)
        for r, s, a in zip(alive, S, acts):
            agree += int(a == teacher(s))
            total += 1
            r.step(int(a))

    crashed = sum(r.crashed for r in robots)
    dist = [sum(math.dist(a, b) for a, b in zip(r.trail, r.trail[1:])) for r in robots]
    print(f"oda {room}: {len(robots)} robot x {args.steps} adim | carpan {crashed} | "
          f"ogretmenle ayni karar {agree / total:.1%} | ort. yol {np.mean(dist):.1f} m | "
          f"adim basina predict {np.median(step_times) * 1000:.0f} ms")
