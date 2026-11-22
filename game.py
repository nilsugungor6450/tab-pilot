"""ZAP! - teach a TabPFN clone by driving, then race it on tracks it has never seen.

Usage:
    python game.py [--mute] [--teach-seconds 40] [--race-seconds 35]
    python game.py --autoplay human --fast --shots shots/
"""
import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import pygame

import bots
import fx
from brain import Brain
from levels import (ROOMS, LapCounter, _to_world, cell_progress, compass, next_target, race_room,
                    teach_room)
from sim import ACTION_NAMES, N_SENSORS, ROBOT_RADIUS, Robot

p = argparse.ArgumentParser()
p.add_argument("--fps", type=int, default=16, help="simulasyon adimi / sn (cizim her zaman 60 fps)")
p.add_argument("--teach-seconds", type=int, default=40)
p.add_argument("--race-seconds", type=int, default=35)
p.add_argument("--hitbox", type=float, default=0.12,
               help="carpisma yaricapi (m). Karakter 0.22 cizilir; kucuk hitbox = kenara surtunmek affedilir")
p.add_argument("--mute", action="store_true", help="sesleri kapat")
p.add_argument("--show-fps", action="store_true", help="kare hizini terminale yaz")
p.add_argument("--autoplay", choices=list(bots.STYLES), default=None,
               help="test icin: oyuncuyu bu bot surer, ekranlar kendiliginden gecer")
p.add_argument("--shots", default=None, help="test icin: ekran goruntulerini bu klasore kaydet")
p.add_argument("--max-races", type=int, default=0, help="test icin: bu kadar yaristan sonra cik")
p.add_argument("--fast", action="store_true", help="test icin: beklemeden kos (her kare = bir adim)")
args = p.parse_args()

W, H, HUD = 1280, 760, 76
SIDE = 360
RENDER_FPS = 60
TICK = 1.0 / args.fps
BG_TOP, BG_BOT = (8, 10, 24), (16, 12, 34)
WIRE, WIRE_GLOW = (140, 240, 255), (40, 140, 255)
ARROW = (34, 48, 84)
PLAYER, CLONE = (255, 196, 70), (186, 120, 255)
TEXT, DIM, GOOD, BAD = (235, 238, 250), (120, 126, 160), (110, 230, 160), (255, 95, 95)
PRETTY = ["Forward", "Slight left", "Slight right", "Sharp left", "Sharp right"]
COLS = [f"s{i}" for i in range(N_SENSORS)] + ["track_sin", "track_cos"]

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("ZAP! - Teach a TabPFN clone")
FONT_NAMES = "avenirnext,helveticaneue,helvetica,arial"
F = {}
for size, bold in [(14, False), (16, False), (20, False), (24, True), (30, True), (44, True), (64, True),
                   (110, True), (160, True)]:
    F[size] = pygame.font.SysFont(FONT_NAMES, size, bold=bold)
MONO = pygame.font.SysFont("menlo,monaco,couriernew", 14)
MONO_B = pygame.font.SysFont("menlo,monaco,couriernew", 14, bold=True)
clock = pygame.time.Clock()
sfx = fx.Sfx(enabled=not args.mute and not args.fast)
particles = fx.Particles()
shake = fx.Shake()
brain = Brain()
if args.shots:
    os.makedirs(args.shots, exist_ok=True)


def features(r):
    return list(r.sense()) + compass(r)


class View:
    def __init__(self, room):
        pts = np.array([pt for seg in ROOMS[room]["walls"] for pt in seg])
        self.min = pts.min(0)
        span = pts.max(0) - self.min
        aw, ah = W - SIDE - 60, H - HUD - 70
        self.s = min(aw / span[0], ah / span[1])
        self.off = np.array([30, 34]) + ((aw, ah) - span * self.s) / 2

    def px(self, x, y):
        return (self.off[0] + (x - self.min[0]) * self.s,
                H - (self.off[1] + (y - self.min[1]) * self.s))


def text(s, pos, size=20, color=TEXT, center=False, target=None, font=None):
    img = (font or F[size]).render(s, True, color)
    r = img.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    (target or screen).blit(img, r)
    return r


def glow_text(s, pos, size, color, strength=2):
    img = F[size].render(s, True, color)
    pad = size // 2
    halo = pygame.Surface((img.get_width() + 2 * pad, img.get_height() + 2 * pad))
    halo.blit(img, (pad, pad))
    halo = fx.blur(halo, 8)
    r = halo.get_rect(center=pos)
    for _ in range(strength):
        screen.blit(halo, r, special_flags=pygame.BLEND_ADD)
    screen.blit(img, img.get_rect(center=pos))


def background():
    surf = pygame.Surface((W, H))
    for y in range(H):
        k = y / H
        surf.fill(tuple(int(a + (b - a) * k) for a, b in zip(BG_TOP, BG_BOT)), (0, y, W, 1))
    for x in range(0, W, 32):
        for y in range(0, H, 32):
            surf.set_at((x, y), (30, 34, 60))
    return surf


BACKGROUND = background()
WORLD = pygame.Surface((W, H))
_TRACK_CACHE = {}


def track_arrows(room):
    g = ROOMS[room]["grid"]
    out = []
    for i, j in np.argwhere(g["free"]):
        cx, cy = _to_world(g, i, j)
        tx, ty = next_target(room, cx, cy, look=1)
        if (tx, ty) != (cx, cy):
            out.append(((cx, cy), math.atan2(ty - cy, tx - cx)))
    return out


def track_surface(view, room):
    if room in _TRACK_CACHE:
        return _TRACK_CACHE[room]
    surf = BACKGROUND.copy()
    for (x, y), a in track_arrows(room):
        px, py = view.px(x, y)
        L = 0.32 * view.s
        dx, dy = math.cos(a), -math.sin(a)
        for k in (-0.5, 0.5):
            tipx, tipy = px + dx * L * (0.5 + k), py + dy * L * (0.5 + k)
            back = 0.45 * L
            pygame.draw.lines(surf, ARROW, False, [
                (tipx - dx * back - dy * back, tipy - dy * back + dx * back), (tipx, tipy),
                (tipx - dx * back + dy * back, tipy - dy * back - dx * back)], 5)
    walls = [(view.px(*a), view.px(*b)) for a, b in ROOMS[room]["walls"]]
    glow_layer = pygame.Surface((W, H))
    for a, b in walls:
        pygame.draw.line(glow_layer, WIRE_GLOW, a, b, 16)
    for blur_k, times in ((14, 2), (5, 1)):
        g = fx.blur(glow_layer, blur_k)
        for _ in range(times):
            surf.blit(g, (0, 0), special_flags=pygame.BLEND_ADD)
    for a, b in walls:
        pygame.draw.line(surf, (70, 170, 230), a, b, 6)
        pygame.draw.line(surf, WIRE, a, b, 3)
    sx, sy, sh = ROOMS[room]["start"]
    c = ROOMS[room]["grid"]["c"]
    nx_, ny_ = -math.sin(sh), math.cos(sh)
    n = 10
    for k in range(n):
        p0 = view.px(sx + nx_ * c * (k / n - 0.5), sy + ny_ * c * (k / n - 0.5))
        p1 = view.px(sx + nx_ * c * ((k + 1) / n - 0.5), sy + ny_ * c * ((k + 1) / n - 0.5))
        pygame.draw.line(surf, (235, 235, 235) if k % 2 == 0 else (60, 60, 70), p0, p1, 6)
    _TRACK_CACHE[room] = (surf, walls)
    return _TRACK_CACHE[room]


def draw_world_track(target, t):
    surf, walls = track_surface(view, room)
    target.blit(surf, (0, 0))
    for i, (a, b) in enumerate(walls):
        L = math.dist(a, b)
        if L < 1:
            continue
        for k in range(max(1, int(L // 260))):
            u = ((t * 140 + i * 97 + k * 260) % L) / L
            pos = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
            fx.blit_glow(target, (120, 220, 255), pos, 12, 0.9)


def interp(r, alpha):
    px_, py_, ph = getattr(r, "prev", (r.x, r.y, r.heading))
    dh = (r.heading - ph + math.pi) % (2 * math.pi) - math.pi
    return px_ + (r.x - px_) * alpha, py_ + (r.y - py_) * alpha, ph + dh * alpha


def step(r, a):
    r.prev = (r.x, r.y, r.heading)
    r.step(a)
    r.near = float(r.sense().min())


def draw_character(target, r, color, alpha, t, ghost=False, label=None, dt=0.0):
    x, y, heading = interp(r, alpha)
    px, py = view.px(x, y)
    rad = max(11, ROBOT_RADIUS * view.s * 1.4)
    near = getattr(r, "near", 5.0)
    scared = near < 0.55 and not r.crashed
    if not r.crashed and dt > 0 and random.random() < 0.9:
        back = heading + math.pi
        particles.emit(px - math.cos(heading) * rad * 0.8, py + math.sin(heading) * rad * 0.8, color,
                       n=1, speed=40, life=0.5, size=3, spread=0.5, angle=-back)
    prev = getattr(r, "disp_heading", heading)
    diff = (heading - prev + math.pi) % (2 * math.pi) - math.pi
    r.disp_heading = prev + min(1.0, 10 * max(dt, 1 / RENDER_FPS)) * diff
    hx, hy = math.cos(r.disp_heading), -math.sin(r.disp_heading)
    if scared:
        px += random.uniform(-1.5, 1.5)
        py += random.uniform(-1.5, 1.5)
    flick = 0.55 + 0.15 * math.sin(t * 23) if ghost else 1.0
    fx.blit_glow(target, color, (px, py), rad * 3.2, 0.55 * flick)
    body = pygame.Surface((int(rad * 2 + 4), int(rad * 2 + 4)), pygame.SRCALPHA)
    c = rad + 2
    base = tuple(int(v * 0.75) for v in color)
    pygame.draw.circle(body, base, (c, c), rad)
    pygame.draw.circle(body, color, (c - rad * 0.15, c - rad * 0.15), rad * 0.8)
    pygame.draw.circle(body, tuple(min(255, v + 60) for v in color), (c - rad * 0.35, c - rad * 0.4), rad * 0.3)
    nx_, ny_ = -hy, hx
    eye = rad * (0.36 if scared else 0.27)
    for side in (-1, 1):
        ex = c + hx * rad * 0.38 + nx_ * side * rad * 0.38
        ey = c + hy * rad * 0.38 + ny_ * side * rad * 0.38
        pygame.draw.circle(body, (255, 255, 255), (ex, ey), eye)
        pygame.draw.circle(body, (18, 16, 36), (ex + hx * eye * 0.35, ey + hy * eye * 0.35), eye * 0.55)
    mx, my = c + hx * rad * 0.7, c + hy * rad * 0.7
    if scared:
        pygame.draw.circle(body, (40, 20, 30), (mx, my), rad * 0.16)
    if ghost:
        body.set_alpha(int(200 * flick))
    target.blit(body, (px - c, py - c))
    if r.crashed:
        for _ in range(4):
            ang = random.uniform(0, 2 * math.pi)
            pts = [(px, py)]
            for k in range(1, 5):
                pts.append((px + math.cos(ang) * k * 10 + random.uniform(-6, 6),
                            py + math.sin(ang) * k * 10 + random.uniform(-6, 6)))
            pygame.draw.lines(target, (255, 250, 180), False, pts, 2)
    if label:
        text(label, (px, py + rad + 16) if ghost else (px, py - rad - 18), 14, color, center=True, target=target)


def zap_burst(r, color):
    x, y = view.px(r.x, r.y)
    particles.emit(x, y, (255, 250, 200), n=40, speed=420, life=0.6, size=4, drag=4)
    particles.emit(x, y, color, n=30, speed=260, life=0.9, size=5, drag=3)
    shake.kick(16)
    sfx.play("zap")


def hud(left, sub, right, color=TEXT, bars=None):
    pygame.draw.rect(screen, (14, 17, 38), (0, 0, W, HUD))
    pygame.draw.line(screen, (50, 60, 110), (0, HUD), (W, HUD), 2)
    text(left, (24, 8), 30, color)
    text(sub, (26, 48), 14, DIM)
    img = F[24].render(right, True, TEXT)
    screen.blit(img, (W - 24 - img.get_width(), 12))
    for k, (val, col) in enumerate(bars or []):
        y = 46 + k * 12
        pygame.draw.rect(screen, (34, 38, 70), (W - 324, y, 300, 6), border_radius=3)
        pygame.draw.rect(screen, col, (W - 324, y, int(300 * min(1.0, max(0.0, val))), 6), border_radius=3)


SHORT = ["Forward", "Left", "Right", "Sharp L", "Sharp R"]


def sidebar():
    x0 = W - SIDE
    pygame.draw.rect(screen, (13, 15, 34), (x0, HUD, SIDE, H - HUD))
    pygame.draw.line(screen, (50, 60, 110), (x0, HUD), (x0, H), 2)
    return x0 + 16, HUD + 16


def steps(lines, x, y, color=DIM):
    for k, s in enumerate(lines):
        text(s, (x, y + k * 22), 16, color)


def table_panel(rows, flash):
    x0, y0 = sidebar()
    w = SIDE - 32
    text("TabPFN's table", (x0, y0), 20, PLAYER)
    text(f"{len(rows)} rows", (x0 + w - 70, y0 + 2), 16, TEXT)
    hdr = f"{'#':>4} {'front':>5} {'left':>4} {'right':>5} {'track':>5}  move"
    screen.blit(MONO.render(hdr, True, DIM), (x0, y0 + 34))
    pygame.draw.line(screen, (50, 60, 110), (x0, y0 + 54), (x0 + w, y0 + 54), 1)
    last = rows[-12:]
    for k, row in enumerate(last):
        idx = len(rows) - len(last) + k + 1
        ang = math.degrees(math.atan2(row[N_SENSORS], row[N_SENSORS + 1]))
        line = f"{idx:>4} {row[0]:5.1f} {row[6]:4.1f} {row[18]:5.1f} {ang:+4.0f}°  {SHORT[ACTION_NAMES.index(row[len(COLS)])]}"
        newest = k == len(last) - 1
        col = PLAYER if newest and flash > 0 else (TEXT if newest else (140, 146, 180))
        screen.blit((MONO_B if newest else MONO).render(line, True, col), (x0, y0 + 62 + k * 19))
    y = y0 + 62 + 12 * 19 + 24
    text("How it works", (x0, y), 20, TEXT)
    steps(["1  You drive. Follow the arrows.",
           "2  Every step = one row: 24 distance",
           "    sensors + track direction + your move",
           "3  TabPFN reads the table in-context.",
           "    No training, 0 gradient steps.",
           "4  Your clone races tracks it never saw."], x0, y + 32)


def brain_panel(probs):
    x0, y0 = sidebar()
    w = SIDE - 32
    text("Clone's brain", (x0, y0), 20, CLONE)
    text("TabPFN's probabilities for the next move", (x0, y0 + 30), 14, DIM)
    best = int(np.argmax(probs)) if probs is not None else -1
    for i, name in enumerate(PRETTY):
        y = y0 + 60 + i * 30
        pr = float(probs[i]) if probs is not None else 0.0
        text(name, (x0, y), 16, TEXT if i == best else DIM)
        bx, bw = x0 + 110, w - 160
        pygame.draw.rect(screen, (40, 36, 72), (bx, y + 5, bw, 12), border_radius=6)
        pygame.draw.rect(screen, CLONE if i == best else (110, 80, 170), (bx, y + 5, int(bw * pr), 12), border_radius=6)
        text(f"{pr:.0%}", (x0 + w - 42, y), 16, TEXT if i == best else DIM)
    y = y0 + 60 + 5 * 30 + 24
    text("The race", (x0, y), 20, TEXT)
    steps(["This track was never in the table.",
           f"The clone only knows {context_rows} rows",
           "of YOUR driving on another track.",
           "Touch the wire and you're out.",
           "Most track progress wins."], x0, y + 32)
    pygame.draw.circle(screen, PLAYER, (x0 + 8, H - 40), 7)
    text("YOU", (x0 + 22, H - 50), 16, PLAYER)
    pygame.draw.circle(screen, CLONE, (x0 + 98, H - 40), 7)
    text("CLONE (TabPFN)", (x0 + 112, H - 50), 16, CLONE)


HOLD_SHARP = 6
held = {"dir": None, "frames": 0}


def player_action(r):
    if args.autoplay:
        return bots.STYLES[args.autoplay](r)
    k = pygame.key.get_pressed()
    d = "L" if (k[pygame.K_LEFT] or k[pygame.K_a]) else "R" if (k[pygame.K_RIGHT] or k[pygame.K_d]) else None
    held["frames"] = held["frames"] + 1 if d is not None and d == held["dir"] else (1 if d else 0)
    held["dir"] = d
    sharp = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or held["frames"] > HOLD_SHARP
    if d == "L":
        return 3 if sharp else 1
    if d == "R":
        return 4 if sharp else 2
    return 0


def new_robot(room_name, x=None, y=None, heading=None):
    r = Robot(room_name, x, y, heading)
    r.hitbox = args.hitbox
    r.near = float(r.sense().min())
    return r


def respawn_point(r):
    p_, cell = cell_progress(r.room, r.x, r.y)
    if p_ is None:
        return None, None, None
    cx, cy = _to_world(ROOMS[r.room]["grid"], *cell)
    tx, ty = next_target(r.room, cx, cy, look=1)
    return cx, cy, math.atan2(ty - cy, tx - cx)


def shot(name):
    if args.shots:
        pygame.image.save(screen, os.path.join(args.shots, f"{name}.png"))


state, ticks = "title", 0
state_time = 0.0
rows, policy, context_note = [], None, ""
race_k, races_done = 0, 0
player = clone = view = room = None
lap_p = lap_c = None
respawn_wait, result, respawn = 0, None, (None, None, None)
context_rows = 0
clone_probs = None
row_flash = 0.0
result_anim = 0.0
seed_base = int(time.time()) % 10_000 * 100
shots_done = set()


def set_state(s):
    global state, ticks, state_time
    state, ticks, state_time = s, 0, 0.0


def set_room(name):
    global room, view
    room, view = name, View(name)


def start_teach(keep=False):
    global rows, player, respawn_wait, lap_p
    set_state("teach")
    respawn_wait = 0
    if not keep:
        rows = []
    set_room(teach_room())
    player, lap_p = new_robot(room), LapCounter(room)
    sfx.play("go")


def start_countdown():
    global player, clone, lap_p, lap_c, clone_probs
    set_state("countdown")
    set_room(race_room(race_k, seed_base))
    player, clone = new_robot(room), new_robot(room)
    lap_p, lap_c = LapCounter(room), LapCounter(room)
    clone_probs = None
    ask_clone()


def ask_clone():
    if not clone.crashed:
        brain.request("probs", np.array([features(clone)]))


def save_rows():
    if not rows:
        return
    os.makedirs("demos/plays", exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS + ["action", "x", "y", "heading"])
    df.to_csv(f"demos/plays/teach_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)


def fallback_rows():
    r, out = Robot(teach_room()), []
    for _ in range(420):
        a = bots.arrow_follower(r)
        out.append(features(r) + [ACTION_NAMES[a]])
        r.step(a)
    return out


def start_loading():
    global context_note
    set_state("loading")
    save_rows()
    use = [r[:len(COLS) + 1] for r in rows]
    if len(use) >= 40 and len({r[-1] for r in use}) >= 2:
        context_note = f"clone of YOU · {min(len(use), 2000)} rows · 0 gradient steps"
    else:
        use = fallback_rows()
        context_note = "not enough rows - clone of the demo driver"
    X = np.array([r[:-1] for r in use], dtype=float)
    y = np.array([r[-1] for r in use])
    brain.request("fit", X, y)
    sfx.play("whoosh")


def sim_tick():
    global respawn_wait, respawn, row_flash, clone_probs, result, races_done, result_anim
    if state == "teach":
        if respawn_wait > 0:
            respawn_wait -= 1
            if respawn_wait == 0:
                globals()["player"] = new_robot(room, *respawn)
                lap_p.last, _ = cell_progress(room, player.x, player.y)
                sfx.play("blip")
            return
        a = player_action(player)
        rows.append(features(player) + [ACTION_NAMES[a], player.x, player.y, player.heading])
        row_flash = 0.12
        step(player, a)
        lap_p.update(player)
        if player.crashed:
            del rows[-int(args.fps):]
            respawn_wait = args.fps
            respawn = respawn_point(player)
            zap_burst(player, PLAYER)
    elif state == "race":
        if not player.crashed:
            step(player, player_action(player))
            lap_p.update(player)
            if player.crashed:
                zap_burst(player, PLAYER)
        if not clone.crashed and brain.busy:
            clone_probs = brain.result()[0]
            step(clone, int(np.argmax(clone_probs)))
            lap_c.update(clone)
            if clone.crashed:
                zap_burst(clone, CLONE)
            ask_clone()
        if ticks >= args.race_seconds * args.fps or (player.crashed and clone.crashed):
            result = (lap_p.laps, lap_c.laps, player.crashed, clone.crashed)
            races_done += 1
            result_anim = 0.0
            set_state("result")
            win = result[0] > result[1] + 0.05
            sfx.play("win" if win else "lose")
            x, y = (W // 2, H // 2 - 110)
            particles.emit(x, y, PLAYER if win else CLONE, n=90, speed=520, life=1.6, size=5, drag=1.5)


def draw_title(t):
    screen.blit(BACKGROUND, (0, 0))
    if random.random() < 0.5:
        particles.emit(random.uniform(0, W), H + 10, random.choice([PLAYER, CLONE, WIRE]), n=1,
                       speed=90, life=4, size=3, spread=0.3, angle=-math.pi / 2, drag=0.1)
    particles.draw(screen)
    wob = math.sin(t * 2) * 4
    glow_text("ZAP!", (W // 2, H // 2 - 150 + wob), 160, WIRE, strength=3)
    text("Don't touch the wire.", (W // 2, H // 2 - 40), 30, TEXT, center=True)
    text(f"Drive laps for {args.teach_seconds} seconds. TabPFN turns your driving into a table",
         (W // 2, H // 2 + 20), 20, DIM, center=True)
    text("and races your clone on tracks it has never seen.  No training. Zero gradient steps.",
         (W // 2, H // 2 + 48), 20, DIM, center=True)
    text("Follow the arrows   ·   LEFT / RIGHT (or A / D) to steer   ·   hold longer to turn sharper",
         (W // 2, H // 2 + 110), 20, TEXT, center=True)
    if int(t * 2) % 2 == 0:
        glow_text("Press SPACE to start", (W // 2, H // 2 + 190), 30, PLAYER, strength=1)


def draw_loading(t):
    screen.blit(BACKGROUND, (0, 0))
    bx, by, bw, bh = W // 2 + 120, H // 2 - 80, 240, 160
    pulse = 0.6 + 0.4 * math.sin(t * 6)
    fx.blit_glow(screen, CLONE, (bx + bw / 2, by + bh / 2), 220, 0.5 * pulse)
    pygame.draw.rect(screen, (24, 20, 52), (bx, by, bw, bh), border_radius=16)
    pygame.draw.rect(screen, CLONE, (bx, by, bw, bh), 3, border_radius=16)
    text("TabPFN", (bx + bw / 2, by + bh / 2 - 12), 44, TEXT, center=True)
    text("in-context learning", (bx + bw / 2, by + bh / 2 + 30), 16, DIM, center=True)
    n = len(rows)
    for k in range(14):
        u = ((t * 0.9 + k / 14) % 1.0)
        x = 140 + (bx - 160) * u
        y = H // 2 + math.sin(k * 1.7) * 140 * (1 - u)
        a = int(255 * (1 - abs(u - 0.5) * 1.6))
        bar = pygame.Surface((90, 12), pygame.SRCALPHA)
        bar.fill((*PLAYER, max(0, min(255, a))))
        screen.blit(bar, (x, y))
    text(f"Loading your {n} rows into TabPFN...", (W // 2, 140), 44, TEXT, center=True)
    text("no training  ·  no fine-tuning  ·  0 gradient steps", (W // 2, 196), 20, DIM, center=True)
    particles.draw(screen)


def draw_race_scene(t, alpha, dt, labels=True):
    world = WORLD
    draw_world_track(world, t)
    draw_character(world, clone, CLONE, alpha, t, ghost=True, label="CLONE" if labels else None, dt=dt)
    draw_character(world, player, PLAYER, alpha, t, label="YOU" if labels else None, dt=dt)
    particles.draw(world)
    ox, oy = shake.offset(dt)
    screen.fill(BG_TOP)
    screen.blit(world, (ox, oy))


def main():
    global row_flash, result_anim, race_k
    t0 = time.time()
    acc = 0.0
    running = True
    while running:
        dt = 1 / RENDER_FPS if args.fast else clock.tick(RENDER_FPS) / 1000
        dt = min(dt, 0.1)
        t = time.time() - t0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if state == "title" and ev.key == pygame.K_SPACE:
                    start_teach()
                elif state == "result" and ev.key == pygame.K_SPACE:
                    race_k += 1
                    start_countdown()
                elif state == "result" and ev.key == pygame.K_t:
                    start_teach(keep=True)
                elif state == "result" and ev.key == pygame.K_n:
                    start_teach(keep=False)

        globals()["state_time"] += dt
        if state in ("teach", "race"):
            acc += TICK if args.fast else dt
            while acc >= TICK and state in ("teach", "race"):
                acc -= TICK
                sim_tick()
                globals()["ticks"] += 1
            alpha = 1.0 if args.fast else acc / TICK
        else:
            acc, alpha = 0.0, 1.0
        particles.update(dt)
        row_flash = max(0.0, row_flash - dt)

        if state == "title":
            draw_title(t)
            if args.autoplay and state_time > 0.1:
                shot("title")
                start_teach()

        elif state == "teach":
            world = WORLD
            draw_world_track(world, t)
            if respawn_wait == 0 or int(t * 10) % 2 == 0:
                draw_character(world, player, PLAYER, alpha, t, label="YOU", dt=dt)
            particles.draw(world)
            ox, oy = shake.offset(dt)
            screen.fill(BG_TOP)
            screen.blit(world, (ox, oy))
            total = args.teach_seconds * args.fps
            left = max(0, (total - ticks) / args.fps)
            hud(f"TEACH  ·  {left:4.1f} s", "follow the arrows  ·  every step you drive becomes one row in TabPFN's table",
                f"{len(rows)} rows   ·   {lap_p.laps:.1f} laps", PLAYER, bars=[(lap_p.laps, PLAYER)])
            table_panel(rows, row_flash)
            if left <= 5:
                text(f"{left:.0f}", ((W - SIDE) // 2, H // 2), 110, (255, 255, 255), center=True)
            if ticks == total // 2 and "teach" not in shots_done:
                shot("teach")
                shots_done.add("teach")
            if ticks >= total:
                start_loading()

        elif state == "loading":
            draw_loading(t)
            if state_time > 0.3 and "loading" not in shots_done:
                shot("loading")
                shots_done.add("loading")
            if brain.ready() and state_time > 1.2:
                globals()["context_rows"] = brain.result()
                race_k = 0
                start_countdown()

        elif state == "countdown":
            draw_race_scene(t, 1.0, 0.0)
            hud(ROOMS[room]["title"], f"never seen by TabPFN  ·  {context_note}", "get ready")
            brain_panel(None)
            k = 3 - int(state_time / 0.7)
            if k != getattr(main, "last_k", None):
                main.last_k = k
                sfx.play("go" if k <= 0 else "beep")
            label = "GO!" if k <= 0 else str(k)
            glow_text(label, ((W - SIDE) // 2, H // 2), 160, PLAYER if k > 0 else GOOD, strength=2)
            if state_time > 2.8:
                set_state("race")

        elif state == "race":
            draw_race_scene(t, alpha, dt)
            total = args.race_seconds * args.fps
            left = max(0, (total - ticks) / args.fps)
            hud(f"{ROOMS[room]['title']}  ·  {left:4.1f} s", f"never seen by TabPFN  ·  {context_note}",
                f"YOU {lap_p.laps:.2f}    CLONE {lap_c.laps:.2f} laps",
                bars=[(lap_p.laps, PLAYER), (lap_c.laps, CLONE)])
            brain_panel(clone_probs)
            key = f"race{races_done}"
            if ticks >= args.fps * 8 and key not in shots_done:
                shot(key)
                shots_done.add(key)

        elif state == "result":
            result_anim = min(1.0, result_anim + dt / 1.0)
            draw_race_scene(t, 1.0, dt, labels=False)
            veil = pygame.Surface((W, H), pygame.SRCALPHA)
            veil.fill((5, 6, 15, 175))
            screen.blit(veil, (0, 0))
            particles.draw(screen)
            lp, lc, pz, cz = result
            if abs(lp - lc) < 0.05:
                banner, col = "IT'S A TIE", TEXT
            elif lp > lc:
                banner, col = "YOU WIN", PLAYER
            else:
                banner, col = "YOUR CLONE WINS", CLONE
            glow_text(banner, (W // 2, H // 2 - 110), 110 if len(banner) < 12 else 64, col, strength=2)
            e = 1 - (1 - result_anim) ** 3
            text(f"YOU  {lp * e:.2f} laps" + ("  ·  zapped" if pz else ""), (W // 2, H // 2), 30, PLAYER, center=True)
            text(f"CLONE  {lc * e:.2f} laps" + ("  ·  zapped" if cz else ""), (W // 2, H // 2 + 46), 30, CLONE, center=True)
            text(f"Your clone drove a track it had never seen  ·  {context_note}",
                 (W // 2, H // 2 + 100), 16, DIM, center=True)
            text("SPACE next track   ·   T teach more (adds rows)   ·   N new player   ·   Esc quit",
                 (W // 2, H // 2 + 150), 20, TEXT, center=True)
            key = f"result{races_done - 1}"
            if state_time > 1.1 and key not in shots_done:
                shot(key)
                shots_done.add(key)
            if args.autoplay and state_time > 1.3:
                if args.max_races and races_done >= args.max_races:
                    running = False
                else:
                    race_k += 1
                    start_countdown()

        pygame.display.flip()
        if args.show_fps and int(t) != int(t - dt):
            print(f"{state:9s} {clock.get_fps():5.1f} fps", flush=True)

    brain.close()
    pygame.quit()


if __name__ == "__main__":
    main()
