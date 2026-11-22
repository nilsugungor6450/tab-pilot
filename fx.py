"""Visual and sound effects: glow sprites, particles, screen shake, synthesized sounds."""
import math
import random

import numpy as np
import pygame

_GLOW = {}


def glow(color, radius, strength=1.0):
    key = (tuple(color), int(radius), round(strength, 1))
    if key not in _GLOW:
        r = max(2, int(radius))
        y, x = np.mgrid[-r:r, -r:r] + 0.5
        d = np.sqrt(x * x + y * y) / r
        a = np.clip(1 - d, 0, 1) ** 2 * strength
        surf = pygame.Surface((2 * r, 2 * r))
        arr = np.zeros((2 * r, 2 * r, 3))
        for k in range(3):
            arr[..., k] = a * color[k]
        pygame.surfarray.blit_array(surf, np.clip(arr, 0, 255).astype(np.uint8).transpose(1, 0, 2))
        _GLOW[key] = surf
    return _GLOW[key]


def blit_glow(target, color, pos, radius, strength=1.0):
    g = glow(color, radius, strength)
    target.blit(g, (pos[0] - g.get_width() / 2, pos[1] - g.get_height() / 2), special_flags=pygame.BLEND_ADD)


def blur(surf, factor=6):
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.transform.smoothscale(small, (w, h))


class Particles:
    def __init__(self):
        self.p = []

    def emit(self, x, y, color, n=1, speed=60, life=0.6, size=4, spread=math.pi, angle=None, drag=2.5):
        for _ in range(n):
            a = (angle if angle is not None else random.uniform(-math.pi, math.pi)) + random.uniform(-spread, spread)
            v = speed * random.uniform(0.3, 1.0)
            L = life * random.uniform(0.6, 1.2)
            self.p.append([x, y, math.cos(a) * v, math.sin(a) * v, L, L, color, size * random.uniform(0.6, 1.3), drag])

    def update(self, dt):
        keep = []
        for q in self.p:
            q[4] -= dt
            if q[4] <= 0:
                continue
            q[0] += q[2] * dt
            q[1] += q[3] * dt
            q[2] *= math.exp(-q[8] * dt)
            q[3] *= math.exp(-q[8] * dt)
            keep.append(q)
        self.p = keep

    def draw(self, target, offset=(0, 0)):
        for x, y, _, _, life, max_life, color, size, _ in self.p:
            k = life / max_life
            blit_glow(target, color, (x + offset[0], y + offset[1]), size * (0.5 + k) * 2, strength=k)


class Shake:
    def __init__(self):
        self.amp = 0.0

    def kick(self, amount):
        self.amp = max(self.amp, amount)

    def offset(self, dt):
        self.amp *= math.exp(-8 * dt)
        if self.amp < 0.3:
            return 0, 0
        return random.uniform(-self.amp, self.amp), random.uniform(-self.amp, self.amp)


class Sfx:
    def __init__(self, enabled=True):
        self.ok = False
        self.s = {}
        if not enabled:
            return
        try:
            pygame.mixer.init(22050, -16, 1, 512)
            self.ok = True
        except pygame.error:
            return
        sr = 22050

        def tone(freqs, dur, vol=0.35, decay=6.0, kind="sine"):
            t = np.arange(int(sr * dur)) / sr
            wave = np.zeros_like(t)
            seg = len(t) // len(freqs)
            for i, f in enumerate(freqs):
                tt = t[: seg]
                w = np.sin(2 * np.pi * f * tt) if kind == "sine" else np.sign(np.sin(2 * np.pi * f * tt)) * 0.5
                wave[i * seg:(i + 1) * seg] = w * np.exp(-decay * tt)
            return wave * vol

        def noise(dur, vol=0.5, decay=9.0):
            t = np.arange(int(sr * dur)) / sr
            n = np.random.uniform(-1, 1, len(t))
            crackle = (np.random.rand(len(t)) < 0.02) * np.random.uniform(-1, 1, len(t)) * 2
            return np.clip((n * 0.6 + crackle) * np.exp(-decay * t), -1, 1) * vol

        sr, _, channels = pygame.mixer.get_init()

        def snd(wave):
            data = (np.clip(wave, -1, 1) * 32000).astype(np.int16)
            if channels > 1:
                data = np.repeat(data[:, None], channels, axis=1)
            return pygame.sndarray.make_sound(np.ascontiguousarray(data))

        try:
            self._build(tone, noise, snd)
        except Exception as e:
            print("sound disabled:", e)
            self.ok, self.s = False, {}

    def _build(self, tone, noise, snd):
        self.s = {
            "zap": snd(noise(0.45) + tone([110, 70], 0.45, vol=0.3, decay=5, kind="square")),
            "beep": snd(tone([880], 0.12, vol=0.3, decay=10)),
            "go": snd(tone([1320], 0.35, vol=0.35, decay=5)),
            "win": snd(tone([523, 659, 784, 1046], 0.6, vol=0.3, decay=4)),
            "lose": snd(tone([392, 330, 262], 0.6, vol=0.3, decay=4)),
            "blip": snd(tone([660, 990], 0.12, vol=0.2, decay=12)),
            "whoosh": snd(noise(0.5, vol=0.15, decay=4)),
        }

    def play(self, name):
        if self.ok and name in self.s:
            self.s[name].play()
