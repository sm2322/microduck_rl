# 00 — Getting started

This repo trains control policies for **Microduck**, a 25 cm, ~800 g bipedal
robot with 14 Dynamixel XL330 servos. You describe a behavior as a reward
function in simulation, train it on a GPU, export it to ONNX, and install it on
a real robot.

Three machines can be involved. You do **not** need all three:

| Job | Machine | GPU? |
|---|---|---|
| Edit rewards, look at poses, play a policy | your laptop | no |
| Train (minutes to hours) | cloud GPU (HF Jobs, Colab) or a local NVIDIA box | yes |
| Run the policy | the robot (or CPU MuJoCo as a rehearsal) | no |

## 1. Install

You need [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
uv sync
```

The first sync pulls PyTorch and CUDA wheels: a few GB, and the default 30 s
HTTP timeout may abort on a slow link. If that happens:

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

`uv sync` is the ground truth. HF Jobs and Colab run it from scratch, so a
package that only exists because you installed it by hand will fail remotely.

## 2. See the four tasks

```bash
uv run list-envs
```

```
Mjlab-Velocity-Flat-MicroDuck    walking, commanded velocity + head pose
Mjlab-StandUp-Flat-MicroDuck     stand up after a fall, then hold
Mjlab-SitStand-Flat-MicroDuck    commanded sit <-> stand
Mjlab-Flamingo-Flat-MicroDuck    one-legged hold (the teaching task)
```

All four share the same observation and action layout, so the robot can
hot-swap their policies without re-reading a schema. See
[01-the-robot.md](01-the-robot.md).

## 3. Look before you train

No GPU needed. This is the fastest way to check that a pose you invented is
physically sensible:

```bash
uv run scripts/view_pose.py --pose home
uv run scripts/view_pose.py --pose flamingo --settle 3
uv run scripts/view_pose.py --pose sit --no-viewer
```

It prints trunk height, both foot heights and trunk tilt. A pose that topples
(>30° tilt, trunk below 7 cm) cannot be a training target. More in
[04-visualize-locally.md](04-visualize-locally.md).

## 4. Smoke test, then train

Always run five iterations at 64 envs first. It costs cents and catches almost
every config error (missing term, wrong dimension, NaN, broken export):

```bash
uv run train Mjlab-Flamingo-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5
```

If that finishes with rising reward and no NaN, launch the real run:

```bash
uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096
```

No local GPU? Add `--hf-jobs` and the same command runs on Hugging Face Jobs —
see [05-train-on-cloud.md](05-train-on-cloud.md). A Colab notebook lives in
[`notebooks/train_colab.ipynb`](../notebooks/train_colab.ipynb).

## 5. Watch, export, deploy

```bash
# watch a run's checkpoint (GPU)
uv run play Mjlab-Flamingo-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# checkpoint -> ONNX, observation normalizer baked in
uv run scripts/export.py Mjlab-Flamingo-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000

# rehearse on CPU with the same actuator model as training
uv run scripts/infer_policy.py --standing output.onnx --new-cmd-obs

# share it with other duck owners
uv run publish --onnx output.onnx --repo <you>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5
```

Deployment details, including `robotctl` commands, are in
[06-deploy-to-robot.md](06-deploy-to-robot.md).

## Where to go next

1. [The robot](01-the-robot.md) — dimensions, joints, actuators, observation contract.
2. [Reward engineering](02-reward-engineering.md) — the rules that make policies transfer.
3. [Flamingo walkthrough](03-flamingo-walkthrough.md) — build your own behavior.
4. [Visualize locally](04-visualize-locally.md)
5. [Train on cloud](05-train-on-cloud.md)
6. [Deploy to robot](06-deploy-to-robot.md)
7. [Troubleshooting](07-troubleshooting.md)
