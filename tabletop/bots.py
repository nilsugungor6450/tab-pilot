"""Scripted drivers used for testing and autoplay."""
import numpy as np

from levels import compass
from sim import _sector, teacher

MIRROR = [(24 - i) % 24 for i in range(24)]
SWAP = [0, 2, 1, 4, 3]


def right_wall(r):
    return teacher(r.sense())


def left_wall(r):
    return SWAP[teacher(r.sense()[MIRROR])]


def arrow_follower(r):
    d = r.sense()
    s, c = compass(r)
    if _sector(d, 0, 20) < 0.9:
        return 3 if _sector(d, 60, 30) > _sector(d, -60, 30) else 4
    if _sector(d, 60, 30) < 0.5:
        return 2
    if _sector(d, -60, 30) < 0.5:
        return 1
    if s > 0.3:
        return 3 if c < 0.2 else 1
    if s < -0.3:
        return 4 if c < 0.2 else 2
    return 0


class Human:
    def __init__(self, seed=0, lag=3, error=0.05):
        self.rng, self.lag, self.error = np.random.default_rng(seed), lag, error
        self.hist, self.cur, self.left = [], 0, 0

    def __call__(self, r):
        self.hist.append((r.x, r.y, r.heading))
        x, y, h = self.hist[max(0, len(self.hist) - 1 - self.lag)]
        if self.left > 0:
            self.left -= 1
            return self.cur
        past = type(r)(r.room, x, y, h)
        a = arrow_follower(past)
        if a in (3, 4):
            self.cur, self.left = a, int(self.rng.integers(0, 3))
        elif a in (1, 2):
            self.cur, self.left = a, int(self.rng.integers(1, 5))
        if self.rng.random() < self.error:
            a = int(self.rng.integers(5))
        return a


STYLES = {"arrows": arrow_follower, "human": Human(0), "right-wall": right_wall, "left-wall": left_wall}
