"""TableTop - canli demo (ekrandaki yazilar Ingilizce).

Kullanim:
    python app.py                          # rastgele, hic gorulmemis bir oda
    python app.py --room B                 # elle cizilmis oda B

Tuslar (ekranda da yazar):
  Kim suruyor?
    P        TabPFN surer (baglamdaki tablo satirlarini taklit eder)
    T        Teacher: kural tabanli kontrolcu surer (TabPFN'in ogrendigi "ogretmen")
    M        Sen surersin: yukari ok = ileri, sol/sag ok = hafif donus,
             Shift + sol/sag = keskin donus. Bastigin her adim bir tablo satiri olur.
  TabPFN'in tablosu (baglam):
    F        "Feed": TabPFN'e SENIN satirlarini ver, artik seni taklit eder
    G        Geri: ogretmen tablosuna don
  Oda / robot:
    N        Yeni rastgele oda (hic gorulmemis)      1 2 3  Oda A / B / C
    R        Robotu basa al                          Fare   Robotu tikladigin yere koy
  Gorunum:
    C        "Where is it going?" belirsizlik konisini ac/kapa
    Esc      Cikis
"""
import argparse
import math
import time

import numpy as np
import pygame

from forecast import HORIZONS, to_world, train_default
from policy import TabPFNPolicy
from sim import ACTION_NAMES, ROOMS, SENSOR_ANGLES, Robot, random_room, teacher

p = argparse.ArgumentParser()
p.add_argument("--room", default=None, help="A, B, C veya R<sayi>; bos = rastgele yeni oda")
p.add_argument("--demos", default="demos/teacher_mixed.csv")
p.add_argument("--fps", type=int, default=10)
p.add_argument("--frames", type=int, default=0, help="N kare sonra kapat (test icin)")
p.add_argument("--screenshot", default=None, help="kapanirken ekran goruntusu kaydet")
p.add_argument("--no-cone", action="store_true", help="konum konisini kapat (daha hizli)")
p.add_argument("--mode", default="tabpfn", choices=["tabpfn", "teacher", "manual"])
p.add_argument("--seed", type=int, default=None, help="N tusu icin rastgele oda baslangici")
args = p.parse_args()

W, H, PANEL = 1180, 720, 400
BG, WALL, TRAIL = (17, 20, 38), (230, 232, 245), (90, 110, 200)
CONE = (255, 205, 90)
ACCENT, OK, BAD, DIM = (124, 140, 255), (110, 220, 160), (255, 110, 110), (130, 135, 160)
BAR_COLORS = [(110, 220, 160), (124, 170, 255), (255, 190, 100), (90, 120, 255), (255, 130, 90)]
PRETTY = ["Forward", "Slight left", "Slight right", "Sharp left", "Sharp right"]

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("TableTop - TabPFN drives a robot")
font = pygame.font.SysFont("Helvetica", 18)
big = pygame.font.SysFont("Helvetica", 30, bold=True)
small = pygame.font.SysFont("Helvetica", 14)
key_font = pygame.font.SysFont("Menlo", 13, bold=True)
clock = pygame.time.Clock()
rng = np.random.default_rng(args.seed)


# ------------------------------------------------------------ yardimcilar
class View:
    """Metre <-> piksel donusumu (odaya gore olcekler)."""

    def __init__(self, room):
        pts = np.array([pt for seg in ROOMS[room]["walls"] for pt in seg])
        self.min = pts.min(0)
        span = pts.max(0) - self.min
        self.s = min((W - PANEL - 60) / span[0], (H - 60) / span[1])
        self.off = np.array([30, 30]) + ((W - PANEL - 60, H - 60) - span * self.s) / 2

    def px(self, x, y):   # y yukari pozitif -> ekranda asagi
        return (self.off[0] + (x - self.min[0]) * self.s,
                H - (self.off[1] + (y - self.min[1]) * self.s))

    def m(self, px, py):
        return ((px - self.off[0]) / self.s + self.min[0],
                (H - py - self.off[1]) / self.s + self.min[1])


def text(s, pos, f=font, color=WALL):
    screen.blit(f.render(s, True, color), pos)


def key(label, desc, pos):
    """Tus kutucugu + aciklama."""
    w = key_font.size(label)[0] + 10
    pygame.draw.rect(screen, (55, 60, 100), (pos[0], pos[1], w, 18), border_radius=4)
    screen.blit(key_font.render(label, True, WALL), (pos[0] + 5, pos[1] + 1))
    text(desc, (pos[0] + w + 6, pos[1]), small, DIM)


def message(s):
    """Uzun islem oncesi ekrana not dus (TabPFN yuklenirken)."""
    pygame.draw.rect(screen, (40, 44, 80), (W - PANEL + 20, H / 2 - 22, PANEL - 40, 44), border_radius=8)
    text(s, (W - PANEL + 34, H / 2 - 10))
    pygame.display.flip()


def load_policy(X, y, label):
    message(f"Loading {len(X)} rows into TabPFN...")
    pol = TabPFNPolicy().fit(X, y)
    return pol, f"{label} ({pol.n_context} rows)"


def load_teacher_rows(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df.drop(columns="action").to_numpy(), df["action"].to_numpy()


def new_random_room():
    return random_room(int(rng.integers(10_000, 1_000_000)))   # egitim odalari R500-R509, cakismaz


def room_label(room):
    return f"Room {room}" + ("  (random, never seen)" if room.startswith("R") else "")


# ------------------------------------------------------------ durum
room = args.room or new_random_room()
view = View(room)
robot = Robot(room)
mode = args.mode
manual_X, manual_y = [], []

screen.fill(BG)
teacher_X, teacher_y = load_teacher_rows(args.demos)
policy, context_label = load_policy(teacher_X, teacher_y, "teacher demos")
forecaster = None
if not args.no_cone:
    message("Loading the 'where is it going?' model...")
    forecaster = train_default()
cone_on, cone = forecaster is not None, None

probs = np.zeros(len(ACTION_NAMES))
chosen, predict_ms, frame = None, 0.0, 0

running = True
while running:
    manual_action = None
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            running = False
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_p:
                mode = "tabpfn"
            elif k == pygame.K_t:
                mode = "teacher"
            elif k == pygame.K_m:
                mode = "manual"
            elif k == pygame.K_r:
                robot = Robot(room)
            elif k == pygame.K_n:
                room = new_random_room()
                view, robot = View(room), Robot(room)
            elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                room = "ABC"[k - pygame.K_1]
                view, robot = View(room), Robot(room)
            elif k == pygame.K_f and len(manual_X) >= 20 and len(set(manual_y)) >= 2:
                policy, context_label = load_policy(np.array(manual_X), np.array(manual_y),
                                                    "YOUR driving")
                mode, robot = "tabpfn", Robot(room)
            elif k == pygame.K_g:
                policy, context_label = load_policy(teacher_X, teacher_y, "teacher demos")
            elif k == pygame.K_c and forecaster is not None:
                cone_on = not cone_on
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[0] < W - PANEL:
            x, y = view.m(*ev.pos)
            robot = Robot(room, x, y, robot.heading)

    if mode == "manual":
        keys = pygame.key.get_pressed()
        shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if keys[pygame.K_LEFT]:
            manual_action = 3 if shift else 1
        elif keys[pygame.K_RIGHT]:
            manual_action = 4 if shift else 2
        elif keys[pygame.K_UP]:
            manual_action = 0

    # ---------------------------------------------------- bir simulasyon adimi
    sensors = robot.sense()
    ts = time.time()
    probs = policy.probs(sensors)[0]
    predict_ms = (time.time() - ts) * 1000
    if robot.crashed:
        chosen = None
    elif mode == "tabpfn":
        chosen = int(probs.argmax())
    elif mode == "teacher":
        chosen = teacher(sensors)
    else:
        chosen = manual_action
        if chosen is not None:
            manual_X.append(sensors)
            manual_y.append(ACTION_NAMES[chosen])
    if chosen is not None:
        robot.step(chosen)

    # "where is it going?" konisi: 1..4 sn sonrasi icin %10 / %50 / %90 tahminleri
    cone = None
    if cone_on and not robot.crashed:
        fwd, lat = forecaster.predict(robot.sense())
        pose = (robot.x, robot.y, robot.heading)
        left = [to_world(pose, f, l) for f, l in zip(fwd[:, 1], lat[:, 2])]    # %90 = en solu
        right = [to_world(pose, f, l) for f, l in zip(fwd[:, 1], lat[:, 0])]   # %10 = en sagi
        mid = [to_world(pose, f, l) for f, l in zip(fwd[:, 1], lat[:, 1])]
        reach = [(to_world(pose, fwd[i, 0], lat[i, 1]), to_world(pose, fwd[i, 2], lat[i, 1]))
                 for i in range(len(HORIZONS))]                              # ileri yonde aralik
        cone = (left, right, mid, reach)

    # ---------------------------------------------------- cizim
    screen.fill(BG)
    rx, ry = view.px(robot.x, robot.y)
    for ang, d in zip(SENSOR_ANGLES, sensors):                    # sensor isinlari
        th = robot.heading + ang
        end = view.px(robot.x + math.cos(th) * d, robot.y + math.sin(th) * d)
        t = min(d / 2.5, 1)
        col = (int(255 - 120 * t), int(90 + 60 * t), int(90 + 120 * t))
        pygame.draw.line(screen, col, (rx, ry), end, 1)
        pygame.draw.circle(screen, col, end, 3)
    if cone is not None:                                          # belirsizlik konisi
        left, right, mid, reach = cone
        poly = [(rx, ry)] + [view.px(*q) for q in left] + [view.px(*q) for q in reversed(right)]
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(layer, (*CONE, 70), poly)
        pygame.draw.polygon(layer, (*CONE, 170), poly, 2)
        for a, b in reach:
            pygame.draw.line(layer, (*CONE, 200), view.px(*a), view.px(*b), 3)
        screen.blit(layer, (0, 0))
        for q in mid:
            pygame.draw.circle(screen, CONE, view.px(*q), 4)
        tip = view.px(*mid[-1])
        text(f"+{HORIZONS[-1]:g} s", (tip[0] + 8, tip[1] - 8), small, CONE)
    if len(robot.trail) > 1:                                      # iz
        pygame.draw.lines(screen, TRAIL, False, [view.px(*q) for q in robot.trail[-1500:]], 2)
    for a, b in ROOMS[room]["walls"]:                             # duvarlar
        pygame.draw.line(screen, WALL, view.px(*a), view.px(*b), 4)
    pygame.draw.circle(screen, BAD if robot.crashed else ACCENT, (rx, ry), 0.22 * view.s)
    hx, hy = view.px(robot.x + math.cos(robot.heading) * 0.35, robot.y + math.sin(robot.heading) * 0.35)
    pygame.draw.line(screen, WALL, (rx, ry), (hx, hy), 3)

    # panel
    x0 = W - PANEL
    pygame.draw.rect(screen, (26, 30, 56), (x0, 0, PANEL, H))
    text("TableTop", (x0 + 24, 18), big)
    text(room_label(room), (x0 + 24, 56), small, DIM)
    mode_txt = {"tabpfn": "TabPFN is driving", "teacher": "Teacher is driving",
                "manual": "You are driving"}[mode]
    text(mode_txt, (x0 + 24, 80), big, OK if mode == "tabpfn" else ACCENT)
    text(f"Context: {context_label}  ·  0 gradient steps", (x0 + 24, 118), small, DIM)
    if mode == "manual":
        ready = len(manual_X) >= 20 and len(set(manual_y)) >= 2
        text(f"Recorded rows: {len(manual_X)}" + ("  ->  press F to teach TabPFN" if ready else
                                                  "  (need 20+, two kinds of moves)"),
             (x0 + 24, 138), small, OK if ready else WALL)

    text("TabPFN: what's the next move?", (x0 + 24, 166), font)
    for i, (name, pr) in enumerate(zip(PRETTY, probs)):
        y = 196 + i * 40
        text(name, (x0 + 24, y), small, WALL if i == chosen else DIM)
        pygame.draw.rect(screen, (45, 50, 85), (x0 + 24, y + 18, PANEL - 100, 12), border_radius=6)
        pygame.draw.rect(screen, BAR_COLORS[i], (x0 + 24, y + 18, max(4, (PANEL - 100) * pr), 12),
                         border_radius=6)
        text(f"{pr:.0%}", (x0 + PANEL - 66, y + 14), small, WALL)
    text(f"decision time: {predict_ms:.0f} ms", (x0 + 24, 398), small, DIM)
    if cone is not None:
        w = math.dist(cone[0][-1], cone[1][-1])
        text("Where is it going? (yellow cone)", (x0 + 24, 424), font, CONE)
        text(f"80% range, {HORIZONS[-1]:g} s ahead: {w:.2f} m wide", (x0 + 24, 448), small, DIM)
    if robot.crashed:
        text("CRASHED!  Press R to restart", (x0 + 24, 472), font, BAD)

    # tus rehberi
    y0 = H - 196
    text("Who drives", (x0 + 24, y0), small, WALL)
    key("P", "TabPFN", (x0 + 24, y0 + 20)); key("T", "Teacher", (x0 + 130, y0 + 20)); key("M", "You", (x0 + 240, y0 + 20))
    text("You drive: Up = forward, Left/Right = turn, +Shift = sharp", (x0 + 24, y0 + 44), small, DIM)
    text("TabPFN's table", (x0 + 24, y0 + 70), small, WALL)
    key("F", "learn from my driving", (x0 + 24, y0 + 90)); key("G", "back to teacher", (x0 + 210, y0 + 90))
    text("World", (x0 + 24, y0 + 116), small, WALL)
    key("N", "new room", (x0 + 24, y0 + 136)); key("1 2 3", "rooms A B C", (x0 + 130, y0 + 136))
    key("R", "restart", (x0 + 24, y0 + 160)); key("C", "cone on/off", (x0 + 130, y0 + 160))
    text("click: move robot", (x0 + 260, y0 + 160), small, DIM)

    pygame.display.flip()
    frame += 1
    if args.frames and frame >= args.frames:
        running = False
    clock.tick(args.fps)

if args.screenshot:
    pygame.image.save(screen, args.screenshot)
pygame.quit()
