"""2D robot simulation: rooms made of wall segments, a robot with 24 range sensors,
and a rule-based right-wall-following teacher.
"""
import math

import numpy as np

N_SENSORS = 24
SENSOR_ANGLES = np.deg2rad(np.arange(N_SENSORS) * 360 / N_SENSORS)
MAX_RANGE = 5.0
ROBOT_RADIUS = 0.22
DT = 0.15
SPEED = 0.45

ACTIONS = [
    ("Move-Forward",       1.0,  0.0),
    ("Slight-Left-Turn",   0.9,  0.9),
    ("Slight-Right-Turn",  0.9, -0.9),
    ("Sharp-Left-Turn",    0.25, 2.6),
    ("Sharp-Right-Turn",   0.35, -2.2),
]
ACTION_NAMES = [a[0] for a in ACTIONS]


def _poly(*pts):
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _box(x, y, w, h):
    return _poly((x, y), (x + w, y), (x + w, y + h), (x, y + h))


ROOMS = {
    "A": {
        "walls": _poly((0, 0), (10, 0), (10, 4), (7, 4), (7, 7), (0, 7)) + _box(3, 2.5, 1.5, 1.5),
        "start": (1.2, 3.0, math.pi / 2),
    },
    "B": {
        "walls": _poly((0, 0), (5, 0), (5, 2), (9, 2), (9, 0), (12, 0), (12, 8), (6, 8), (6, 5), (0, 5))
                 + _box(9, 5, 1, 1),
        "start": (1.0, 2.5, math.pi / 2),
    },
    "C": {
        "walls": _poly((0, 0), (11, 0), (11, 3), (8, 3), (8, 6), (11, 6), (11, 9), (0, 9), (0, 6),
                       (3, 6), (3, 3), (0, 3)),
        "start": (1.0, 1.5, 0.0),
    },
}


def random_room(seed):
    name = f"R{seed}"
    if name in ROOMS:
        return name
    rng = np.random.default_rng(seed)
    nx, ny = rng.integers(4, 8), rng.integers(3, 6)
    c = rng.uniform(1.8, 2.8)
    filled = np.zeros((nx, ny), bool)
    target = int(nx * ny * rng.uniform(0.5, 0.8))
    filled[rng.integers(nx), rng.integers(ny)] = True
    while filled.sum() < target:
        cells = np.argwhere(filled)
        i, j = cells[rng.integers(len(cells))]
        di, dj = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.integers(4)]
        if 0 <= i + di < nx and 0 <= j + dj < ny:
            filled[i + di, j + dj] = True

    def f(i, j):
        return 0 <= i < nx and 0 <= j < ny and filled[i, j]

    segs = []
    for j in range(ny + 1):
        run = None
        for i in range(nx + 1):
            edge = i < nx and f(i, j - 1) != f(i, j)
            if edge and run is None:
                run = i
            if not edge and run is not None:
                segs.append(((run * c, j * c), (i * c, j * c)))
                run = None
    for i in range(nx + 1):
        run = None
        for j in range(ny + 1):
            edge = j < ny and f(i - 1, j) != f(i, j)
            if edge and run is None:
                run = j
            if not edge and run is not None:
                segs.append(((i * c, run * c), (i * c, j * c)))
                run = None

    cells = [tuple(x) for x in np.argwhere(filled)]
    si, sj = cells[rng.integers(len(cells))]
    pillar_size = c - 1.5
    if pillar_size >= 0.35:
        for (i, j) in cells:
            if (i, j) != (si, sj) and rng.random() < 0.25:
                s = rng.uniform(0.35, pillar_size)
                x0, y0 = (i + 0.5) * c - s / 2, (j + 0.5) * c - s / 2
                segs += _box(x0, y0, s, s)

    th = rng.uniform(0, 2 * math.pi)
    cx, cy = nx * c / 2, ny * c / 2
    cos, sin = math.cos(th), math.sin(th)

    def rot(p):
        x, y = p[0] - cx, p[1] - cy
        return (cx + cos * x - sin * y, cy + sin * x + cos * y)

    ROOMS[name] = {
        "walls": [(rot(p), rot(q)) for p, q in segs],
        "start": (*rot(((si + 0.5) * c, (sj + 0.5) * c)), rng.uniform(-math.pi, math.pi)),
    }
    return name


def wall_array(walls):
    return np.array([[p[0], p[1], q[0], q[1]] for p, q in walls], dtype=float)


def raycast(x, y, heading, walls_np, angles=SENSOR_ANGLES, max_range=MAX_RANGE):
    th = heading + angles
    dx, dy = np.cos(th)[:, None], np.sin(th)[:, None]
    x1, y1, x2, y2 = (walls_np[:, i][None, :] for i in range(4))
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((x1 - x) * ey - (y1 - y) * ex) / denom
        u = ((x1 - x) * dy - (y1 - y) * dx) / denom
    hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


def dist_to_walls(x, y, walls_np):
    x1, y1, x2, y2 = walls_np.T
    ex, ey = x2 - x1, y2 - y1
    L2 = ex * ex + ey * ey
    s = np.clip(((x - x1) * ex + (y - y1) * ey) / L2, 0, 1)
    px, py = x1 + s * ex, y1 + s * ey
    return np.sqrt((x - px) ** 2 + (y - py) ** 2).min()


class Robot:
    def __init__(self, room, x=None, y=None, heading=None):
        if room not in ROOMS and room.startswith("R"):
            random_room(int(room[1:]))
        self.room = room
        self.walls = wall_array(ROOMS[room]["walls"])
        sx, sy, sh = ROOMS[room]["start"]
        self.x = sx if x is None else x
        self.y = sy if y is None else y
        self.heading = sh if heading is None else heading
        self.crashed = False
        self.trail = [(self.x, self.y)]
        self.hitbox = ROBOT_RADIUS

    def sense(self):
        return raycast(self.x, self.y, self.heading, self.walls)

    def step(self, action):
        _, v_mul, omega = ACTIONS[action]
        h = self.heading + omega * DT
        nx = self.x + math.cos(h) * SPEED * v_mul * DT
        ny = self.y + math.sin(h) * SPEED * v_mul * DT
        self.heading = (h + math.pi) % (2 * math.pi) - math.pi
        if dist_to_walls(nx, ny, self.walls) < self.hitbox:
            self.crashed = True
            return
        self.x, self.y = nx, ny
        self.trail.append((nx, ny))
        if len(self.trail) > 4000:
            self.trail.pop(0)


def _sector(d, deg, width=20):
    diff = (np.rad2deg(SENSOR_ANGLES) - deg + 180) % 360 - 180
    return d[np.abs(diff) <= width].min()


TARGET = 0.75


def teacher(d):
    front = _sector(d, 0, 20)
    front_right = _sector(d, -45, 10)
    right = _sector(d, -90, 20)
    front_left = _sector(d, 40, 25)
    left = _sector(d, 90, 30)
    if front < 0.9:
        return 4 if min(front_left, left) < 0.5 and right > min(front_left, left) else 3
    if front_left < 0.6 or left < 0.45:
        return 4 if min(front_left, left) < 0.4 else 2
    if right > 1.5 and front_right > 1.5:
        if _sector(d, -135, 25) < 1.5:
            return 4
        return 0
    if right < TARGET - 0.15 or front_right < 0.8:
        return 1
    if right > TARGET + 0.15:
        return 2
    return 0
