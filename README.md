# Microduck RL — club edition

Train control policies for [Microduck](https://github.com/pollen-robotics/microduck),
a ~25 cm, ~800 g bipedal robot with 14 Dynamixel XL330 servos.

You write a behavior as a reward function in simulation. The repo trains it with
PPO on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), exports it to
ONNX, and the robot's runtime runs it at 50 Hz. Policies are hot-swappable: every
task in this repo shares the same observation layout.

This is the **club edition**: four tasks, a worked teaching example, and the
minimum machinery to edit a reward, see the robot on a laptop, train on a cloud
GPU, and deploy to hardware.

## What you need

| job | machine | GPU |
|---|---|---|
| edit rewards, inspect poses, replay a policy | any laptop | no |
| train | cloud (HF Jobs / Colab) or a local NVIDIA box | yes |
| run the policy | the robot, or CPU MuJoCo as rehearsal | no |

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
UV_HTTP_TIMEOUT=600 uv sync     # first sync downloads CUDA/torch wheels
uv run list-envs
```

## The four tasks

| task id | what it learns |
|---|---|
| `Mjlab-Velocity-Flat-MicroDuck` | walking with velocity + head-pose commands (the base recipe) |
| `Mjlab-StandUp-Flat-MicroDuck` | stand up after a fall, then hold the stand |
| `Mjlab-SitStand-Flat-MicroDuck` | commanded sit ↔ stand in one policy |
| `Mjlab-Flamingo-Flat-MicroDuck` | one-legged hold — the worked teaching example |

Every task shares the **61-D actor observation** (48 proprioception + twist 3 +
head pose 4 + body pose 6) and the **14-servo action layout**, so the runtime can
hot-swap their policies. Envs that do not use a command slot zero-pad it; they
never delete it.

## Quickstart

```bash
# 1. Look at a pose on CPU (no GPU, no training)
uv run scripts/view_pose.py --pose flamingo --settle 3

# 2. Smoke test: 5 iterations at 64 envs catches almost every config error
uv run train Mjlab-Flamingo-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5

# 3. Train (locally on a GPU, or add --hf-jobs for Hugging Face Jobs)
uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096

# 4. Watch a checkpoint
uv run play Mjlab-Flamingo-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# 5. Export (observation normalizer baked in — mandatory path)
uv run scripts/export.py Mjlab-Flamingo-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000

# 6. Rehearse on CPU with the same BAM actuator model as training
uv run scripts/infer_policy.py --standing output.onnx --new-cmd-obs

# 7. Publish for other duck owners
uv run publish --onnx output.onnx --repo <you>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5
```

No local GPU:

```bash
uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

See [05-train-on-cloud.md](docs/05-train-on-cloud.md) and
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Write your own behavior

Start from the Flamingo walkthrough:
[docs/03-flamingo-walkthrough.md](docs/03-flamingo-walkthrough.md). The short
version:

1. Pick the closest task and copy its env cfg; keep the inherited domain
   randomization, observation noise and delays.
2. Put the target pose in `src/mjlab_microduck/robot/poses.py`, then look at it:
   `uv run scripts/view_pose.py --pose <name> --settle 3`.
3. Add rewards in `src/mjlab_microduck/tasks/mdp.py`, grouped by task. Read
   [docs/02-reward-engineering.md](docs/02-reward-engineering.md) first — the
   sign convention, gates and anti-jackpot rules are not optional.
4. Register the task in `src/mjlab_microduck/tasks/__init__.py`.
5. Add a cfg test (copy `tests/test_flamingo_cfg.py`).
6. Smoke test, then train. Watch the video, not just the reward.

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF models (walk + ground-contact), scenes
│   ├── microduck_constants.py        # robot configs, HOME frame, BAM actuator
│   └── poses.py                      # named poses shared by envs and the viewer
├── actuator/friction_dr_bam.py       # BAM actuator + friction/voltage domain randomization
├── tasks/
│   ├── __init__.py                   # task registry (four tasks)
│   ├── mdp.py                        # all custom rewards, events, observations, curricula
│   ├── symmetry.py                   # optional 61-D mirror loss
│   └── microduck_*_env_cfg.py       # one config per task family
├── export.py                         # checkpoint → ONNX, normalizer baked in
├── publish/                          # manifest builder + Hub upload
├── train_cli.py, train_hook.py       # `train` entry point, --hf-jobs interception
└── hf_jobs.py                        # Hugging Face Jobs submission

scripts/
├── view_pose.py                      # CPU pose viewer / stability check
├── infer_policy.py                   # CPU ONNX playback with BAM actuators
├── export.py                         # CLI wrapper for the export path
└── hf/                               # HF Jobs helpers (README, uploader)
```

## The parts that must not change

- **Observation layout** is 61-D and shared. Do not reorder, do not drop a
  command slot; zero-pad unused slots.
- **Joint layout** is left leg (0–4), neck/head (5–8), right leg (9–13). Use the
  `_servo_joint_ids` / `_servo_joint_pos` helpers in `mdp.py`, not hard-coded
  indices.
- **Actuators are BAM** (voltage-controlled XL330). Joint-friction
  randomization scales the actuator's `friction_scale`; `dof_frictionloss` is
  zeroed under BAM and randomizing it is a silent no-op.
- **The normalizer is baked into the ONNX** by `scripts/export.py`. Never
  hand-convert a checkpoint.
- **Policies are unfiltered** — no action low-pass in training. Adding one
  without a matching runtime flag breaks transfer.
- **Domain randomization must not accumulate across resets.** mjlab's `dr.*`
  ops are non-accumulating; custom randomizers must restore defaults first.

[AGENTS.md](AGENTS.md) is the condensed playbook for AI agents and contributors
working in this repo.

## Publish a policy

`uv run publish` produces the exact layout the robot daemon loads: `policy.onnx`
(normalizer baked in), a schema-2 `manifest.json`, and a README with install
commands.

```bash
# episodic skill
uv run publish --task Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> \
    --repo <you>/microduck-my-walk --kind episodic --duration-s 4.0

# gait for a slot
uv run publish --onnx walk.onnx --repo <you>/microduck-my-walk \
    --kind perpetual --slot walk

# held pose (Flamingo): unwind before handing back to the gait
uv run publish --onnx flamingo.onnx --repo <you>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5
```

The publisher refuses anything that is not `[1,61] → [1,14]`, refuses NaN or
constant outputs, and records the training run in the manifest. Phase-driven and
posture-flag behaviors (SitStand) are driven by the daemon and are not publishable
this way. Full details: [docs/06-deploy-to-robot.md](docs/06-deploy-to-robot.md).

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config and reward-function regression tests. They lock joint indices,
reward signs, NaN guards, the Flamingo cfg, and the aarch64 torch routing.

## Documentation

- [00 — Getting started](docs/00-getting-started.md)
- [01 — The robot](docs/01-the-robot.md)
- [02 — Reward engineering](docs/02-reward-engineering.md)
- [03 — Flamingo walkthrough](docs/03-flamingo-walkthrough.md)
- [04 — Visualize locally](docs/04-visualize-locally.md)
- [05 — Train on cloud](docs/05-train-on-cloud.md)
- [06 — Deploy to robot](docs/06-deploy-to-robot.md)
- [07 — Troubleshooting](docs/07-troubleshooting.md)

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — robot, runtime, `robotctl`
- [mjlab](https://github.com/mujocolab/mjlab) — training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — actuator models

## License

Code: Apache 2.0 — see [LICENSE](LICENSE).

The 3D model files (`.stl` meshes in
`src/mjlab_microduck/robot/microduck/assets/`) are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See
[LICENSE-MODELS.md](LICENSE-MODELS.md) for the exact scope and attribution.
