"""Tracks for the ZAP! game: ASCII levels, random ring tracks, track progress,
the track-direction compass feature and a lap counter.

Level legend: "#" free cell, "o" free cell with a pillar, "> < ^ v" start cell and heading, "." wall.
"""
import math

import numpy as np

from sim import ROOMS, _box

HEADINGS = {">": 0.0, "^": math.pi / 2, "<": math.pi, "v": -math.pi / 2}

TEACH_TRACK = {
    "name": "Training track",
    "cell": 2.7,
    "map": """
##>####...
#.....#...
#.....####
#........#
###......#
..########
""",
}

RACE_LEVELS = [
    {"name": "Warm-up", "cell": 2.8, "map": """
##>####
#.....#
#.....#
#######
"""},
    {"name": "Zigzag", "cell": 2.6, "map": """
#>##.###.
#..#.#.#.
#..###.##
#.......#
#########
"""},
    {"name": "Pillar hall", "cell": 2.6, "map": """
###o>###
####o###
##....##
##....##
#o######
#####o##
"""},
    {"name": "Tight squeeze", "cell": 2.2, "map": """
#>##.###
#..#.#.#
##.###.#
.#.....#
.#######
"""},
]


def walls_from_mask(free, c):
    nx, ny = free.shape

    def f(i, j):
        return 0 <= i < nx and 0 <= j < ny and free[i, j]

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
    return segs


def _register(name, free, c, start_cell, heading, pillars=(), title=None, rotate=0.0):
    segs = walls_from_mask(free, c)
    for (i, j) in pillars:
        s = 0.5
        segs += _box((i + 0.5) * c - s / 2, (j + 0.5) * c - s / 2, s, s)
    nx, ny = free.shape
    cx, cy = nx * c / 2, ny * c / 2
    cos, sin = math.cos(rotate), math.sin(rotate)

    def rot(p):
        x, y = p[0] - cx, p[1] - cy
        return (cx + cos * x - sin * y, cy + sin * x + cos * y)

    si, sj = start_cell
    ROOMS[name] = {
        "walls": [(rot(p), rot(q)) for p, q in segs],
        "start": (*rot(((si + 0.5) * c, (sj + 0.5) * c)), heading + rotate),
        "title": title or name,
        "grid": {"free": free, "c": c, "cx": cx, "cy": cy, "rot": rotate,
                 "progress": progress_field(free, start_cell, heading)},
    }
    return name


NEIGH = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def progress_field(free, start, heading):
    nx, ny = free.shape
    di, dj = round(math.cos(heading)), round(math.sin(heading))
    pi_, pj = -dj, di
    si, sj = start
    cross = [(si, sj)]
    for s in (1, -1):
        k = 1
        while 0 <= si + s * k * pi_ < nx and 0 <= sj + s * k * pj < ny and free[si + s * k * pi_, sj + s * k * pj]:
            cross.append((si + s * k * pi_, sj + s * k * pj))
            k += 1
    barrier = {((i, j), (i - di, j - dj)) for i, j in cross}
    prog = np.full(free.shape, -1)
    queue = list(cross)
    for cell in cross:
        prog[cell] = 0
    while queue:
        i, j = queue.pop(0)
        for a, b in NEIGH:
            n = (i + a, j + b)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n] or prog[n] >= 0:
                continue
            if ((i, j), n) in barrier or (n, (i, j)) in barrier:
                continue
            prog[n] = prog[i, j] + 1
            queue.append(n)
    return prog


def _to_grid(g, x, y):
    cos, sin = math.cos(g["rot"]), math.sin(g["rot"])
    dx, dy = x - g["cx"], y - g["cy"]
    gx, gy = g["cx"] + cos * dx + sin * dy, g["cy"] - sin * dx + cos * dy
    return int(math.floor(gx / g["c"])), int(math.floor(gy / g["c"]))


def _to_world(g, i, j):
    cos, sin = math.cos(g["rot"]), math.sin(g["rot"])
    x, y = (i + 0.5) * g["c"] - g["cx"], (j + 0.5) * g["c"] - g["cy"]
    return g["cx"] + cos * x - sin * y, g["cy"] + sin * x + cos * y


def cell_progress(room, x, y):
    g = ROOMS[room]["grid"]
    i, j = _to_grid(g, x, y)
    nx, ny = g["free"].shape
    if 0 <= i < nx and 0 <= j < ny and g["progress"][i, j] >= 0:
        return int(g["progress"][i, j]), (i, j)
    return None, None


def track_length(room):
    return int(ROOMS[room]["grid"]["progress"].max()) + 1


def next_target(room, x, y, look=2):
    g = ROOMS[room]["grid"]
    prog, free = g["progress"], g["free"]
    N = track_length(room)
    p, cell = cell_progress(room, x, y)
    if p is None:
        return x, y
    nx, ny = free.shape
    for _ in range(look):
        best = None
        for a, b in NEIGH:
            n = (cell[0] + a, cell[1] + b)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and prog[n] >= 0:
                step = (prog[n] - p) % N
                if step in (1, 2) and (best is None or step < best[0]):
                    best = (step, n)
        if best is None:
            break
        cell, p = best[1], int(prog[best[1]])
    return _to_world(g, *cell)


def compass(robot):
    tx, ty = next_target(robot.room, robot.x, robot.y)
    a = math.atan2(ty - robot.y, tx - robot.x) - robot.heading
    return [math.sin(a), math.cos(a)]


class LapCounter:
    def __init__(self, room):
        self.room, self.N = room, track_length(room)
        self.last, _ = cell_progress(room, *ROOMS[room]["start"][:2])
        self.cells = 0

    def update(self, robot):
        p, _ = cell_progress(self.room, robot.x, robot.y)
        if p is None or self.last is None:
            return
        d = (p - self.last) % self.N
        if d > self.N // 2:
            d -= self.N
        self.cells += d
        self.last = p

    @property
    def laps(self):
        return max(0.0, self.cells / self.N)


def ascii_level(level, key):
    if key in ROOMS:
        return key
    rows = [r for r in level["map"].strip("\n").split("\n")]
    w = max(len(r) for r in rows)
    rows = [r.ljust(w, ".") for r in rows]
    ny = len(rows)
    free = np.zeros((w, ny), bool)
    pillars, start, heading = [], None, 0.0
    for r, line in enumerate(rows):
        j = ny - 1 - r
        for i, ch in enumerate(line):
            if ch in "#o><^v":
                free[i, j] = True
            if ch == "o":
                pillars.append((i, j))
            if ch in HEADINGS:
                start, heading = (i, j), HEADINGS[ch]
    return _register(key, free, level["cell"], start, heading, pillars, level["name"])


def random_track(seed):
    name = f"T{seed}"
    if name in ROOMS:
        return name
    rng = np.random.default_rng(seed)
    nx, ny = int(rng.integers(7, 11)), int(rng.integers(5, 8))
    solid = np.zeros((nx, ny), bool)
    x0, y0 = 0, 0
    for k in range(int(rng.integers(1, 4))):
        w, h = int(rng.integers(3, nx + 1)), int(rng.integers(3, ny + 1))
        if k > 0:
            x0 = int(rng.integers(max(0, x0 - w + 2), min(nx - w, x0 + 2) + 1)) if nx - w >= 0 else 0
            y0 = int(rng.integers(max(0, y0 - h + 2), min(ny - h, y0 + 2) + 1)) if ny - h >= 0 else 0
        solid[x0:x0 + w, y0:y0 + h] = True
    pad = np.pad(solid, 1)
    inner = solid.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            inner &= pad[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]
    free = solid & ~inner
    c = float(rng.uniform(2.3, 2.9))
    def ok(i, j):
        return 0 <= i < nx and 0 <= j < ny and free[i, j]

    starts = []
    for i, j in np.argwhere(free):
        for (di, dj), h in [((1, 0), 0.0), ((0, 1), math.pi / 2), ((-1, 0), math.pi), ((0, -1), -math.pi / 2)]:
            if ok(i + di, j + dj) and ok(i - di, j - dj) and not ok(i - dj, j + di) and not ok(i + dj, j - di):
                starts.append(((int(i), int(j)), h))
    (si, sj), heading = starts[int(rng.integers(len(starts)))]
    return _register(name, free, c, (si, sj), heading, title=f"Random track #{seed}",
                     rotate=float(rng.uniform(0, 2 * math.pi)))


def teach_room():
    return ascii_level(TEACH_TRACK, "L_teach")


def race_room(k, seed_base=0):
    if k < len(RACE_LEVELS):
        return ascii_level(RACE_LEVELS[k], f"L{k + 1}")
    return random_track(seed_base + k)
