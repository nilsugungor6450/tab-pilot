# TabPilot

**The robot's brain is a table.** TabPilot drives robots with [TabPFN](https://github.com/PriorLabs/TabPFN), a foundation model for tabular data. Every moment of driving becomes one row: 24 distance sensors plus the move that was made. TabPFN reads those rows as *context* and decides the next move. There is no training loop and no gradient steps: change the rows and you change the behaviour.

Built for the HackLab Berlin track *"TabPFN for spatial awareness"*. The sensor layout and moves come from the real [UCI Wall-Following Robot Navigation](#dataset-uci-wall-following-robot-navigation) dataset.

It has three parts:

- **ZAP!** (`game.py`): an arcade game. You drive for 40 seconds, TabPFN turns your driving into a clone, and the clone races you on tracks it has never seen.
- **Lab** (`app.py` and the experiment scripts): a rule-based teacher, random rooms, an uncertainty cone that shows where the robot is going, and generalization tests.
- **ROS 2 driver** (`ros/tabpfn_driver.py`): the same idea on a real robot software stack. It drives the **FastBot** robot in Gazebo on [The Construct](https://app.theconstruct.ai), using its 360° LiDAR.

---

## Quick start

TabPFN needs Python 3.10–3.14. On Python 3.15, `tabpfn-client` fails to install.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install tabpfn tabpfn-client pygame-ce matplotlib pandas
```

Create an API key at [platform.priorlabs.ai](https://platform.priorlabs.ai) and accept the model license. Then put the key in `.env`:

```bash
echo 'TABPFN_TOKEN=<your-key>' > .env
```

The local `tabpfn` package downloads the model weights on its first run.

Run everything from the repository root:

```bash
python game.py
```

```bash
python app.py
```

---

## How it works

```
sensors (24 distances) + track direction (sin, cos)  ->  one row  ->  TabPFN context
                                                                        |
new situation (same 26 columns)  ->  TabPFN.predict_proba  ->  probabilities for 5 moves  ->  robot moves
```

- **Robot** (`sim.py`): a circle with 24 range sensors, one every 15° up to 5 m, like the UCI Wall-Following robot. It has 5 discrete moves: forward, slight left/right and sharp left/right.
- **Row:** the 24 sensor readings and, in the game, 2 extra columns that give the direction the track continues in relative to the robot. The label is the move that was made.
- **Policy** (`policy.py`): `TabPFNClassifier.fit()` loads the rows as context. Nothing is trained. Each step, `predict_proba` returns a probability for every move, and the robot takes the most likely one. With a cached context and a single estimator, one decision takes about 40 ms on an M2 MacBook Air (8 GB).
- **Separate process** (`brain.py`): in the game, TabPFN runs in its own process. Running it in a thread inside the 60 fps game loop made both slow because of Python's GIL. Loading went from about 5 s to more than 80 s.

### Controls

**ZAP!**

| Key | Action |
|---|---|
| Left / Right (or A / D) | Steer. Hold the key to turn sharper. |
| Space | Start, or go to the next track |
| T | Teach more (adds rows to the table) |
| N | New player (clears the table) |
| Esc | Quit |

Useful flags: `--mute`, `--teach-seconds 30`, `--race-seconds 25`, `--hitbox 0.12`.

**Lab** (`app.py`)

| Key | Action |
|---|---|
| P / T / M | TabPFN, the teacher or you drive |
| F | Load your rows into TabPFN |
| G | Go back to the teacher rows |
| C | Toggle the uncertainty cone |
| N | New random room |
| 1 / 2 / 3 | Rooms A / B / C |
| R | Restart |
| Click | Move the robot to the clicked point |

---

## ROS 2 on The Construct (FastBot)

`ros/tabpfn_driver.py` is a single-file ROS 2 node for running TabPFN on **FastBot in a Gazebo simulation on The Construct**. It doesn't depend on any other file in this repository, so you can copy it into your rosject and run it directly.

How it works:

- The LiDAR scan (`/scan`) is reduced to the same 24 sensors as in the simulator. Each sensor is the median of the beams within ±3° of its direction.
- Moves are sent as `Twist` messages on `/cmd_vel`.
- The moves are the same 5 classes as in the game.

In a terminal where ROS 2 is sourced, inside your rosject:

```bash
pip install tabpfn-client
```

```bash
export TABPFN_TOKEN=<your-key>
```

Check the sensors:

```bash
python3 tabpfn_driver.py --mode probe
```

Record teacher rows (the table):

```bash
python3 tabpfn_driver.py --mode teacher --record rows.csv --steps 600
```

Let TabPFN drive with those rows as context:

```bash
python3 tabpfn_driver.py --mode tabpfn --context rows.csv
```

Notes:

- The driver uses `tabpfn-client` (cloud) when it is installed, and the local `tabpfn` package otherwise. With the cloud client, each decision takes about 1–2 s, so the robot stops before every move ("stop, think, go"). For the demo, I sped up the recording.
- If your topics have different names (check with `ros2 topic list`), pass `--scan-topic` and `--cmd-topic`.
- If left and right look swapped in `--mode probe`, use `--flip`. If the LiDAR is mounted rotated, use `--lidar-offset`.
- `--behavior wander` (the default) drives around without touching anything. `--behavior wall` follows the right wall.

---

## File structure

```
.
├── README.md
├── game.py            ZAP!: teach a clone in 40 s, race it on unseen tracks
├── app.py             Lab: TabPFN / teacher / you drive, uncertainty cone, random rooms
│
├── sim.py             2D simulation: rooms, robot, 24 sensors, 5 moves, right-wall teacher
├── levels.py          Game tracks (ASCII levels, random rings), track progress, compass feature, lap counter
├── policy.py          TabPFN classifier used as the driving policy
├── brain.py           Runs TabPFN in a separate process for the game
├── bots.py            Scripted drivers for testing and --autoplay
├── fx.py              Glow, particles, screen shake, synthesized sounds
│
├── record.py          Record teacher demonstrations as a CSV table
├── evaluate.py        TabPFN drives several robots in unseen rooms: crashes and agreement with the teacher
├── generalize.py      30 unseen random rooms: single-room vs mixed-room context, saves figures/
├── forecast.py        "Where is it going?": TabPFNRegressor quantiles + calibration check
│
├── data/
│   ├── sensor_readings_24.data   UCI Wall-Following dataset, 24 sensors (see Dataset)
│   └── sensor_readings_4.data    Same rows, 4 simplified distances
├── demos/
│   ├── teacher_A.csv       Teacher rows from room A only
│   ├── teacher_mixed.csv   Teacher rows from room A + 10 random rooms (default context)
│   └── plays/              Rows recorded from real ZAP! play sessions
├── figures/
│   ├── generalize_teacher_A.png       TabPFN in 30 unseen rooms, single-room context
│   └── generalize_teacher_mixed.png   Same rooms, mixed-room context
├── docs/
│   └── demo_script.md     3-minute demo script
└── ros/
    └── tabpfn_driver.py   ROS 2 node: FastBot on The Construct (Gazebo) driven by TabPFN
```

---

## Dataset: UCI Wall-Following Robot Navigation

The project started from a real robot dataset:

> Freire, A., Veloso, M., & Barreto, G. (2009). *Wall-Following Robot Navigation Data* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57C8W
> License: CC BY 4.0. Page: https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data

A SCITOS G5 robot followed the walls of a room clockwise for 4 rounds. It had 24 ultrasound sensors arranged in a ring around its body, one every 15°, sampled 9 times per second. The dataset has 5456 rows. Each row is labeled with the move the robot made: `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn` or `Slight-Left-Turn`. There are also simplified versions with 4 distances (front, left, right, back) and with 2 distances.

**Why we chose it**

- **It is spatial awareness in its simplest real form.** A robot senses distances around itself and decides where to go next. That is the track's question, with real sensor data.
- **It is already a table.** One row is one moment: sensor readings plus a move. That is exactly the input TabPFN is built for, and 5456 rows fit in its context.
- **It is time-ordered.** Because the rows are sampled at 9 Hz in sequence, we can ask "what will the robot do in 1, 2 or 5 seconds?" and not only "what is it doing now?".
- **It is known to be hard for simple models.** The dataset description notes that the classes are not linearly separable: a Perceptron fails, while neural networks succeed. That gives a meaningful linear baseline to compare against.
- **It shaped our simulator.** The 24 sensors every 15°, the discrete move classes and the wall-following teacher in `sim.py` copy this setup. Our simulator is what lets us go beyond one recorded room: new rooms, a live robot and a game.

**Files in `data/`**

| File | Columns |
|---|---|
| `sensor_readings_24.data` | 24 ultrasound readings + move |
| `sensor_readings_4.data` | 4 simplified distances (front, left, right, back) + move |

Both files have 5456 rows, no header, and the move as the last column.

---

## Results

All numbers were measured in this repository using the local `tabpfn` package.

**Memorization or generalization?** (`python generalize.py`)

TabPFN drove 30 random rooms it had never seen, 600 steps each. Each room had different corridor widths and pillars, and was rotated by a random angle.

| Context | Crashes | Right wall kept at 0.5–1.0 m |
|---|---|---|
| Teacher (reference) | 0 / 30 | 74% |
| Rows from room A only | 24 / 30 | 72% |
| Rows from room A + 10 random rooms | **0 / 30** | 75% |

The model stayed the same; only the table changed. See the two images in `figures/`.

**Uncertainty cone** (`python forecast.py`): the 10–90% range that TabPFNRegressor predicts for the robot's position 1–4 s ahead contains the true position 65–87% of the time in unseen rooms. The target is 80%.

**Clones of real players**: a clone built from one 40-second ZAP! session raced 12 tracks it had never seen. Most clones had 0–2 crashes and a median of about 0.6 laps in 35 s, faster than the player was while teaching. Some sessions produce a weak clone. Teaching a second round (key T) helps.

---

## Limitations

- The policy has no memory, so it sometimes loops around a pillar.
- Moves are discrete (5 classes), and everything except the ROS driver is a 2D simulation.
- The context is capped at 2000 rows.
- We have not compared against a nearest-neighbour baseline yet.
