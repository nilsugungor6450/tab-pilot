# TabPilot — 3-minute demo script

**Core message:** The robot's brain is a table. No training, no gradients: TabPFN reads demonstration rows as *context* and drives in environments it has never seen.

**Two parts:** (1) ZAP! — a pygame game where you teach your own clone in 40 seconds. (2) The Construct — the same idea in ROS 2 + Gazebo on the FastBot robot, with a real LiDAR.

---

## Setup (before the demo)

| | What | Command / state |
|---|---|---|
| 1 | Game window open on the title screen | `python game.py --teach-seconds 30 --race-seconds 25` |
| 2 | FastBot Gazebo world open on The Construct, two terminals sourced | (launch the FastBot simulation from your The Construct project) |
| 3 | Teacher rows recorded in advance (don't risk it live) | `python3 tabpfn_driver.py --mode teacher --record rows.csv --steps 600` |
| 4 | **Backup video:** sped-up recording of TabPFN driving (cloud call ≈ 1–2 s per step) | — |
| 5 | Backup image: `figures/generalize_teacher_mixed.png` | — |

> Note: by default the game is 40 s teach + 35 s race = 75 s. The shorter flags above keep the whole demo within 3 minutes.

---

## Flow

### 0:00 – 0:20 · Intro (game title screen)

> "Normally, to teach a robot to drive, you train a neural network for hours. We don't. We write every moment of the robot as a row in a table: 24 distance sensors and the action taken at that moment. We hand that table to TabPFN — a foundation model pre-trained for tables. No training, zero gradient steps. Let me show you with a game first."

### 0:20 – 1:20 · ZAP! — "Teach your clone, then race it"

**[SPACE → TEACH screen]** Drive along the arrows, don't touch the wire.

> "Right now I'm driving. The character always moves forward; I only steer left and right. Top right, the row count goes up: every frame is one row — 24 sensors, the direction of the track, and my key press. If I touch the wire — zap — the last second of rows is deleted, so the clone doesn't learn my mistakes."

**[Time's up → "Loading your driving into TabPFN…" screen]**

> "Now these few hundred rows go into TabPFN as context. Look: 'no training, 0 gradient steps'. It takes less than a second."

**[RACE screen — unseen track]**

> "TabPFN has never seen this track. Yellow is me, the purple ghost is my clone. At the bottom you can see what the clone is thinking — 'Sharp left, 87%'. It's a probability, so we also know how confident the model is."

**[RESULT screen]**

> "My clone [won / came close]. Thirty seconds of play turned into a driver — without any training."

### 1:20 – 2:30 · The Construct — same brain, real robot stack

**[The Construct browser tab: FastBot in Gazebo]**

> "Now let's move the same idea onto a real robot software stack. This is ROS 2 and Gazebo on The Construct; the robot is FastBot. The sensor is no longer rays I drew — it's a real 360-degree LiDAR."

**[Terminal: briefly show `scan_to_24` in `tabpfn_driver.py`]**

> "It's a single Python file. We reduce the LiDAR's 360 beams to the same 24 sensors as in the game — the median of ±3 degrees around each direction, to fight noise. The actions are the same five classes: forward, slight left, slight right, sharp left, sharp right."

**[Terminal 2: `python3 tabpfn_driver.py --mode tabpfn --context rows.csv`]**

> "First, a rule-based teacher followed the right wall and recorded rows. We deliberately made it act randomly 20% of the time, so the table also contains 'how do I recover after a mistake' rows. Now TabPFN is driving."

**[Log lines scrolling: `step 12: Slight-Right-Turn (91%, 1.3 s)`]**

> "Each step, the log shows the decision, the confidence and the time. Because TabPFN runs in the cloud, the robot does 'stop, think, go' — I've sped the video up. But it follows the wall — and there's no `train()` anywhere."

### 2:30 – 3:00 · Close (backup image: grid of 30 rooms)

> "So — memorisation or generalisation? We tested in 30 random rooms it had never seen. Learning from a single room, it finished 6 of 30 without a crash. With rows from a mix of rooms in the context: 30 of 30. We didn't change the model — only the table. Programming robot behaviour can be as easy as editing a table. Thank you!"

---

## If something goes wrong

- **TabPFN won't load / is slow:** If there are fewer than 40 rows, the game automatically falls back to the demo driver's rows ("clone of the demo driver"). Say so and carry on.
- **Cloud latency (The Construct):** Start the live run, show 2–3 steps, then switch to the sped-up video.
- **The robot starts circling a pillar:** Known behaviour — the "max-spin" lock forces it forward after 8 sharp turns in a row. Be honest: "It has no memory, so it sometimes loops; the next step is adding past steps to each row."
- **The clone loses in the game:** "More data = better clone" → press `T` to teach a bit more; the rows accumulate.
