# 07 — Troubleshooting

## Install / environment

**`uv sync` hangs or dies downloading CUDA wheels.**
The default HTTP timeout is 30 s. Retry with `UV_HTTP_TIMEOUT=600 uv sync`.
On ARM (DGX Spark, Jetson) the first sync is several GB.

**`torch.cuda.device_count() == 0` on linux-aarch64, then `IndexError` in
`select_gpus()`.**
PyPI's torch wheel for aarch64 is CPU-only. `pyproject.toml` routes torch to
the cu129 index for that platform. If it still happens, check that the direct
`torch==2.9.1` dependency is still present — removing it makes the source
routing a silent no-op. `tests/test_aarch64_cuda_torch.py` locks this.

**`import mjlab_microduck` fails on a fresh sync.**
Run `uv sync` again, then `uv run python -c "import mjlab_microduck"`. A package
installed by hand locally is invisible to HF Jobs and Colab; add it to
`pyproject.toml` instead.

**No wheel for your Python version.**
Use the project interpreter (`uv run …`). `requires-python` is `>=3.12,<3.13`
because BAM pins that range. A system Python 3.9 cannot install MuJoCo.

## Training

**Smoke test never starts: unrecognized `--hf-jobs`.**
The flag is intercepted from the `mjlab.tasks` entry point, not from the `train`
script. Make sure `mjlab_microduck` is installed in the venv
(`uv run python -c "import mjlab_microduck.tasks"`) and that you used
`uv run train`, not a shell alias to some other `train`.

**Loss/obs becomes NaN in the first iterations.**
Check the run tag: `nan_state` should terminate the episode. If the process
still dies, a critic observation term with a sensor-derived value is not
guarded. The velocity recipe wires `_safe` critic terms; envs built standalone
must port them. `tests/test_nan_guard.py` and `tests/test_obs_nan_guard.py`
cover the helpers.

**Reward rises but the behavior never happens.**
Look at `Episode_Reward/<main task term>`, not the total. Total reward can rise
on regularizers alone. If the main term is flat at zero, the Gaussians are too
tight or the target state is unreachable.

**A penalty is positive in wandb.**
Sign error. A self-negating microduck penalty takes a **positive** weight; an
mjlab cost takes a **negative** one. Every `Episode_Reward/<penalty>` must be
`<= 0`. See [02-reward-engineering.md](02-reward-engineering.md).

**The policy does something clever and wrong** (hops instead of lifts a foot,
leans on the lifted foot, head-tripod).
That is reward hacking, not a bug. Add a hard state gate (support contact,
orientation axis, latch) or make the goal multiplicative; do not just raise a
small penalty.

**It learned the start of the move but never the end.**
The frontier gets no on-policy data. Add reverse-curriculum spawns: start
episodes partway through the maneuver, including nearly done.

**A metric steps down exactly at a curriculum boundary.**
The pacing is wrong. Stretch the stage or introduce the harder condition later,
never earlier.

## Poses and rewards

**The robot never reaches the target pose.**
Look at it:

```bash
uv run scripts/view_pose.py --pose <name> --settle 3
```

A pose that topples, or a target height measured on an older robot model, makes
the reward impossible. Re-measure the height in sim; never carry it across model
revisions.

**`scripts/view_pose.py` prints "joint … not in this model".**
The pose has a typo or a joint that does not exist in the selected scene. Every
pose key must be a real MJCF joint name; `tests/test_flamingo_cfg.py` checks the
shared poses.

**The policy parks on a joint limit.**
The stock `dof_pos_limits` cost only fires in the last ~7.5 % of range. Add a
qpos-side limit-proximity penalty on the offending joints.

## Deployment

**The button does nothing / the policy stands still.**
Check the command bytes. SitStand reads a flag in `twist[0]` (`1` = sit); all
zeros means "stand". Flamingo expects all zeros. A walking policy with a zero
command stands. This is the most common "policy ignores the button" cause.

**Works in `play`, fails on the robot.**
`play` applies the observation normalizer itself. Deploy only the ONNX from
`scripts/export.py` or `uv run publish`, which bake it into the graph. Test the
exact file in `scripts/infer_policy.py`.

**Works in `infer_policy.py --no-bam`, fails with BAM.**
BAM is the point: the real XL330 is voltage-controlled with friction. `--no-bam`
is for comparison only.

**Walks in sim, falls on hardware.**
In order: (1) is the exported file the one you meant, from the right
checkpoint? (2) does the runtime feed the same command convention? (3) is the
IMU mounting bias calibrated (DR trains tolerance to magnitude, not a
systematic offset)? (4) is the floor/grippiness outside the training range?
Changing the reward to fix a calibration problem makes it worse.

**HF Jobs submission fails with `403` on job listing.**
Token scope, not billing. The token must be allowed to list jobs in the chosen
namespace. A namespace can also be out of Jobs credits — the error message says
which.

**Colab disconnects mid-run.**
Free sessions are time-limited. Checkpoints are in `logs/`; reconnect, mount
Drive or the HF Hub, and resume with `--agent.load-checkpoint … --agent.resume
True`. For anything longer than an hour, use HF Jobs with `--detach`.

## Tests

```bash
uv run --with pytest pytest tests/
```

They are CPU-only. If a cfg test fails after you deleted or renamed a task, the
registry (`src/mjlab_microduck/tasks/__init__.py`) or a test import still points
at it. If a test asserts on a pyproject entry point or task id, update it with
the change — these tests are the contract.
