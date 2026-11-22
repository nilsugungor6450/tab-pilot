#!/usr/bin/env python3
"""TabPilot on ROS 2: TurtleBot3'u TabPFN ile surer (The Construct / Gazebo).

Tek dosya, paket kurmaya gerek yok. ROS 2 ortami source edilmis bir terminalde:

  1) Ogretmen sursun, satirlari kaydet (TabPFN'in tablosu):
       python3 tabpfn_driver.py --mode teacher --record rows.csv --steps 600
  2) TabPFN sursun (ayni satirlar baglam olur, egitim yok):
       python3 tabpfn_driver.py --mode tabpfn --context rows.csv

LiDAR (360 isin) -> bizim simdeki gibi 24 sensor (15 derecede bir, en fazla 5 m).
Hareketler simdekiyle ayni 5 sinif. TabPFN bulutta calistigi icin (tabpfn-client,
~1-2 sn), robot her karardan once durur: "dur-dusun-git". Videoyu sonra hizlandir.
"""
import argparse
import csv
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

N_SENSORS, MAX_RANGE = 24, 5.0
ANGLES = np.arange(N_SENSORS) * 360 / N_SENSORS
ACTIONS = [("Move-Forward", 1.0, 0.0), ("Slight-Left-Turn", 0.9, 0.9), ("Slight-Right-Turn", 0.9, -0.9),
           ("Sharp-Left-Turn", 0.25, 2.6), ("Sharp-Right-Turn", 0.35, -2.2)]
NAMES = [a[0] for a in ACTIONS]


def scan_to_24(msg, flip=False, offset=0.0):
    """LaserScan -> 24 mesafe. Her yon icin +-3 derecedeki isinlarin medyani (gurultuye karsi).
    flip: LiDAR ters (saat yonunde) olcuyorsa sag-sol duzelt. offset: LiDAR'in robota gore donuklugu (derece)."""
    r = np.array(msg.ranges, dtype=float)
    r[~np.isfinite(r)] = MAX_RANGE
    r = np.clip(r, 0.0, MAX_RANGE)
    ang = np.rad2deg(msg.angle_min + np.arange(len(r)) * msg.angle_increment)
    if flip:
        ang = -ang
    ang = (ang + offset) % 360
    out = []
    for a in ANGLES:
        diff = np.abs((ang - a + 180) % 360 - 180)
        sel = r[diff <= 3]
        out.append(float(np.median(sel)) if len(sel) else MAX_RANGE)
    return np.array(out)


def sector(d, deg, width):
    diff = (ANGLES - deg + 180) % 360 - 180
    return d[np.abs(diff) <= width].min()


def teacher(d, s):
    """Simdeki sag-duvar ogretmeninin aynisi; s = mesafe olcegi (kucuk odalar icin < 1)."""
    front, fr, right = sector(d, 0, 20), sector(d, -45, 10), sector(d, -90, 20)
    fl, left = sector(d, 40, 25), sector(d, 90, 30)
    if front < 0.9 * s:
        return 4 if min(fl, left) < 0.5 * s and right > min(fl, left) else 3
    if fl < 0.6 * s or left < 0.45 * s:
        return 4 if min(fl, left) < 0.4 * s else 2
    if right > 1.5 * s and fr > 1.5 * s:
        return 4 if sector(d, -135, 25) < 1.5 * s else 0
    if right < 0.6 * s or fr < 0.8 * s:
        return 1
    if right > 0.9 * s:
        return 2
    return 0


def wander(d, s):
    """Gezgin ogretmen: duz git, bir sey yaklasinca acik tarafa don, hicbir seye degme.
    Duvar takibinin aksine tek basina duran engellerin etrafinda tur atmaz."""
    front = sector(d, 0, 25)
    fl, fr = sector(d, 45, 20), sector(d, -45, 20)
    left, right = sector(d, 90, 25), sector(d, -90, 25)
    if front < 0.7 * s:
        left_open = np.mean(d[1:12]) > np.mean(d[13:24])
        return 3 if left_open else 4
    if min(fl, left) < 0.35 * s:
        return 4
    if min(fr, right) < 0.35 * s:
        return 3
    if fl < 0.6 * s or left < 0.4 * s:
        return 2
    if fr < 0.6 * s or right < 0.4 * s:
        return 1
    return 0


TEACHERS = {"wall": teacher, "wander": wander}


def load_tabpfn(path):
    """Once bulut (tabpfn-client), yoksa yerel tabpfn."""
    X, y = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if row[0] == "s0":
                continue
            X.append([float(v) for v in row[:-1]])
            y.append(row[-1])
    X, y = np.array(X), np.array(y)
    if len(X) > 2000:
        idx = np.linspace(0, len(X) - 1, 2000).astype(int)
        X, y = X[idx], y[idx]
    try:
        from tabpfn_client import TabPFNClassifier
        backend = "tabpfn-client (cloud)"
    except ImportError:
        from tabpfn import TabPFNClassifier
        backend = "tabpfn (local)"
    model = TabPFNClassifier().fit(X, y)
    return model, backend, len(X)


class Driver(Node):
    def __init__(self, a):
        super().__init__("tabpfn_driver")
        self.a, self.scan = a, None
        self.pub = self.create_publisher(Twist, a.cmd_topic, 10)
        self.create_subscription(LaserScan, a.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.rows = []
        self.rng = np.random.default_rng(0)
        self.model = None
        if a.mode == "tabpfn":
            self.model, backend, n = load_tabpfn(a.context)
            self.get_logger().info(f"TabPFN ready: {backend}, {n} context rows, 0 gradient steps")

    def on_scan(self, msg):
        self.scan = msg

    def drive(self, action, seconds):
        _, mul, omega = ACTIONS[action]
        tw = Twist()
        tw.linear.x = self.a.speed * mul
        tw.angular.z = omega * self.a.turn
        end = time.time() + seconds
        while time.time() < end:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.02)

    def stop(self):
        """Robotu durdur. Simulator son komutu tekrarladigi icin birkac kez sifir hiz gonder."""
        for _ in range(5):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self):
        if self.a.mode == "move":
            tw = Twist()
            tw.linear.x, tw.angular.z = self.a.lin, self.a.ang
            end = time.time() + self.a.seconds
            while time.time() < end:
                self.pub.publish(tw)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.stop()
            self.get_logger().info(f"moved: linear {self.a.lin} m/s, angular {self.a.ang} rad/s, {self.a.seconds} s")
            return
        while self.scan is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        spin = force = 0
        for step in range(self.a.steps):
            rclpy.spin_once(self, timeout_sec=0.05)
            d = scan_to_24(self.scan, self.a.flip, self.a.lidar_offset)
            if self.a.mode == "probe":
                self.get_logger().info(
                    f"FRONT {sector(d, 0, 15):.2f}   LEFT {sector(d, 90, 15):.2f}   "
                    f"RIGHT {sector(d, -90, 15):.2f}   BACK {sector(d, 180, 15):.2f}   (m)")
                time.sleep(1.0)
                continue
            if self.a.mode == "teacher":
                act = TEACHERS[self.a.behavior](d, self.a.scale)
                self.rows.append([*d, NAMES[act]])
                if self.rng.random() < self.a.noise:
                    act = int(self.rng.integers(len(ACTIONS)))
            else:
                self.stop()
                t = time.time()
                p = self.model.predict_proba(d[None])[0]
                act = NAMES.index(self.model.classes_[int(p.argmax())])
                self.get_logger().info(f"step {step}: {NAMES[act]} ({p.max():.0%}, {time.time() - t:.1f} s)")
                spin = spin + 1 if act in (3, 4) else 0
                if self.a.max_spin and spin > self.a.max_spin:
                    self.get_logger().warn("spinning too long, forcing forward")
                    force, spin = 3, 0
                if force > 0:
                    act, force = 0, force - 1
            self.drive(act, self.a.dt)
        self.stop()

    def save_rows(self):
        """Ctrl+C ile durdurulsa bile kaydedilen satirlari yaz."""
        if self.a.record and self.rows:
            with open(self.a.record, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"s{i}" for i in range(N_SENSORS)] + ["action"])
                w.writerows(self.rows)
            self.get_logger().info(f"{len(self.rows)} rows -> {self.a.record}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "move", "teacher", "tabpfn"], required=True)
    p.add_argument("--record", default=None, help="teacher modunda satirlari bu CSV'ye yaz")
    p.add_argument("--context", default="rows.csv", help="tabpfn modunda baglam CSV'si")
    p.add_argument("--behavior", choices=["wall", "wander"], default="wander",
                   help="teacher davranisi: wander = dolas, hicbir seye degme; wall = sag duvari takip et")
    p.add_argument("--noise", type=float, default=0.2,
                   help="teacher modunda adimlarin bu kadarinda rastgele hareket (toparlanma ornekleri icin)")
    p.add_argument("--max-spin", type=int, default=8,
                   help="tabpfn modunda art arda bu kadar keskin donusten sonra duz git (0 = kapali)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.4, help="her kararin uygulandigi sure (sn)")
    p.add_argument("--speed", type=float, default=0.15, help="ileri hiz m/s (TurtleBot3 max ~0.22)")
    p.add_argument("--turn", type=float, default=0.6, help="donus hizlarini olcekle")
    p.add_argument("--scale", type=float, default=0.6, help="ogretmen mesafe esiklerini olcekle")
    p.add_argument("--lin", type=float, default=0.0, help="move modu: ileri hiz (m/s)")
    p.add_argument("--ang", type=float, default=0.0, help="move modu: donus hizi (rad/s, + = sola)")
    p.add_argument("--seconds", type=float, default=1.0, help="move modu: sure (sn)")
    p.add_argument("--flip", action="store_true", help="LiDAR'in sag-solu ters ise")
    p.add_argument("--lidar-offset", type=float, default=0.0, help="LiDAR robota gore donuk takiliysa (derece)")
    p.add_argument("--scan-topic", default="/scan", help="LiDAR konusu (ros2 topic list ile bak)")
    p.add_argument("--cmd-topic", default="/cmd_vel", help="hareket komutu konusu")
    a = p.parse_args()
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:
        rclpy.init()
    node = Driver(a)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.save_rows()
        try:
            node.stop()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
