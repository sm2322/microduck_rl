# AGENTS.md

RL training environments for Microduck — a ~800 g, ~25 cm bipedal robot with 14
Dynamixel XL330 servos — built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp) with PPO (rsl_rl). Policies train at 50 Hz, export to ONNX, and are
deployed by the runtime in `pollen-robotics/microduck`. Sim2real transfer is the
point: every convention below exists because breaking it produced a policy that
worked in the viewer and failed on hardware.

**Club edition scope**: four tasks (Velocity, StandUp, SitStand, Flamingo), flat
terrain only, two robot models, no backlash/roller variants. Do not reintroduce
deleted task families without a reason and the matching tests.

## Commands

```bash
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096    # train (add --hf-jobs for HF Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — always first
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/view_pose.py --pose flamingo --settle 3          # CPU pose check
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>       # → ONNX (normalizer baked in)
uv run scripts/infer_policy.py --standing out.onnx --new-cmd-obs  # CPU deployment rehearsal
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
uv run --with pytest pytest tests/
```

A 5-iteration smoke test at 64 envs catches ~95 % of config errors for cents.
Never launch a long run without one.

## Repo map

- `src/mjlab_microduck/tasks/mdp.py` — ALL custom MDP functions (rewards, events,
  observations, commands, curricula). Add new functions here, grouped by task.
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — one cfg per task family.
  `microduck_velocity_env_cfg.py` is the main walking recipe AND the shared base
  (robot, DR, obs, noise, delays, command slots); Flamingo builds on it.
- `src/mjlab_microduck/tasks/__init__.py` — task registration.
- `src/mjlab_microduck/robot/microduck_constants.py` — robot cfgs, HOME frame,
  BAM actuator cfg.
- `src/mjlab_microduck/robot/poses.py` — named poses shared by envs and
  `scripts/view_pose.py`. Dependency-free: keep it importable without mjlab.
- `src/mjlab_microduck/robot/microduck/` — MJCF exports from Onshape
  (onshape-to-robot, one `config_mjcf_*.json` per model) + scenes.
  `robot_walk.xml` = feet-only collisions; `robot_groundcontact.xml` = curated
  floor-contact set (legs, trunk shells, head, jaw, battery, hips). Both have the
  same 14 servo joints in the same order.
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM actuator + friction
  and voltage DR.
- `src/mjlab_microduck/export.py` — the ONNX export (normalizer baked in);
  `scripts/export.py` wraps it.
- `src/mjlab_microduck/publish/` — `uv run publish`: schema-2 manifest builder +
  ONNX shape/smoke gate + Hub upload. Only constant-command episodic/perpetual
  policies are publishable; posture-flag/phase behaviors are the daemon's.
- `scripts/view_pose.py` — plain MuJoCo pose viewer/stability check, CPU only.
- `scripts/infer_policy.py` — CPU ONNX playback with BAM actuators. It still
  carries legacy flags for deleted tasks; the four-task path is
  `--walking/--standing/--sitstand` with `--new-cmd-obs`.
- `tests/` — cfg-invariant and mdp-function regression tests (CPU, no GPU).

## Invariants — do not break these

- **Obs layout is 61D (actor) and shared**: 48 base proprioception + 13D command
  block `[twist(3), head_pose(4), body_pose(6)]`, in that order. An env that
  doesn't use a command slot ZERO-PADS it (keep the obs term, sample tiny
  ranges) — never delete a slot.
- **Joint layout** (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9–13 right
  leg. Never hardcode joint indices in mdp functions; use `_servo_joint_ids` /
  `_servo_joint_pos`. Unactuated joints are named `passive_*`; selectors use
  `^(?!passive_).*`.
- **Actuators are BAM** (voltage-controlled XL330, friction computed by the
  actuator). Standalone env cfgs must register the
  `expand_bam_friction_fields` startup event, and joint-friction DR must scale
  the actuator's `friction_scale` — `dof_frictionloss` is zeroed under BAM, so
  randomizing it is a silent no-op.
- **Obs normalization is ON** → the normalizer must be baked into the ONNX.
  `scripts/export.py` does this; in-sim `play` hides the bug, so never
  hand-convert a checkpoint.
- **Policies are UNFILTERED** (no action low-pass in training). Don't add EMA
  filtering without a matched runtime flag and a transfer test.
- **Domain randomization must not accumulate across resets.** mjlab's `dr.*` ops
  are non-accumulating; custom DR must restore-then-apply.
- If an obs is remapped to a sensor view (encoder bias, IMU), any REWARD on the
  same quantity must measure the same view, or the policy is punished for
  correcting what it sees.
- **Poses are measured, not guessed.** Target heights come from sim on the
  current model revision. Look at a pose with `scripts/view_pose.py` before
  training on it.
- **Sign convention is testable.** Every `Episode_Reward/<penalty>` in wandb must
  be `<= 0`.

## Building a new env — the workflow

1. **Pick the closest template** and build on it, don't start from scratch:
   locomotion → velocity recipe; episodic pose → StandUp or Flamingo. Building on
   `make_microduck_velocity_env_cfg` keeps DR / obs / noise / delays / NaN guards
   in sync for free. A standalone cfg must port the whole stack yourself (grep
   for what velocity wires: `_safe` critic obs terms, `nan_state` with
   `sensor_names`, `expand_bam_friction_fields`, encoder bias, IMU
   misalignment).
2. **Verify physics assumptions in sim BEFORE training:**
   - A target pose must be a stable equilibrium: hold ctrl for 3 s from noisy
     inits and check TILT, not just height.
   - Measure target heights off the actual robot in sim.
   - Check the ground-contact model's behavior for any task where the body can
     touch the floor.
3. **Config conventions**: `ENABLE_*` toggles + tuned constants at the top of the
   cfg; factory `make_..._env_cfg(play: bool)`; register in `tasks/__init__.py`;
   own `RslRl...RunnerCfg` with a distinct `experiment_name`. Optional symmetry
   mirror loss lives in `symmetry.py` (61-D table); OFF by default, never for
   asymmetric tasks.
4. **Write cfg tests** (copy `tests/test_flamingo_cfg.py`): joint indices resolve,
   reward weights have the intended sign, gates open/closed where expected, pose
   data matches the reward target. These run on CPU.
5. **Smoke test**: builds, steps NaN-free, obs is 61D, every reward term
   computes, ONNX exports.
6. Train, watch the log, and expect 2–5 iterations of reward-hacking
   whack-a-mole.

## Reward design — rules that were each learned the hard way

- **Sign convention:** mjlab-base cost functions return ≥ 0 → negative weight.
  Self-negating microduck functions (`*_penalty`, `*_l1` returning ≤ 0) →
  POSITIVE weight. A negative weight on a self-negating penalty double-negates
  into a reward for the violation, and the policy farms it. **Every
  `Episode_Reward/<penalty>` must be ≤ 0.**
- **RL optimizes the letter of the reward.** Every under-specified degree of
  freedom will be exploited. Encode the maneuver with hard state-based gates
  (support contact, orientation-axis checks, latches), not small nudges.
- **No jackpots:** rate-limit or slew any "reach X" reward. Never gate a positive
  reward on being in a bad state. Use potential-based shaping (pay Δprogress).
- **Multiplicative composites beat additive sums at goal states**, but pick stds
  wide enough that the current policy scores visibly.
- **Regularizers come in two kinds.** Motion-blockers (body_ang_vel,
  angular_momentum, tight pose std) penalize what a dynamic motion requires —
  keep them LOW. Smoothness (action_rate, joint_torque_rate) damps jitter — safe,
  but introduce AFTER skill discovery (curriculum from ~0), or "do nothing" wins.
- **Compare reward mass, not weights**, when copying regularizers between envs.
- **Tracking std** ≈ the error you still care about. Ask whether the error is
  escapable or inherent (the head MUST oscillate while walking). Price only the
  escapable part (e.g. L1 on a 1 s EMA).
- **Joints parking on hard limits:** add a qpos-side limit-proximity penalty on
  the offending joints; the stock `dof_pos_limits` only fires in the last ~7.5 %
  of range, and command-side penalties don't work (wide ctrlrange is
  intentional).

## Commands, observations, dead weights

- **A command input that is never non-zero has dead weights forever.** Every
  command slot keeps a small non-zero sampling range from step 0 (even at reward
  weight 0).
- **Zero-command behavior must be explicitly trained** (`zero_command_prob`-style
  exact-zero sampling): uniform sampling essentially never produces the all-zero
  command, which is the deployment idle state.
- Rare-but-important command regions need explicit buckets (e.g.
  turn-in-place).
- A posture flag is a command. SitStand puts it in `twist[0]` (0 = stand,
  1 = sit); Flamingo uses all-zero commands. Document any new convention.

## Curricula

- Steps are env steps: `iteration × 24`.
- Use `microduck_mdp.reward_weight` for weight schedules (a step function, not an
  interpolation — discretize ramps) and a params-curriculum for command/event
  ranges.
- Mutate term cfgs via the managers (`env.event_manager.get_term_cfg(...)`),
  never `env.cfg.events[...]` — managers deepcopy their cfg at init, so writes to
  `env.cfg` are silent no-ops.
- Phase-align stages with what the policy has learned. If a metric steps DOWN at
  a boundary, the pacing is wrong: stretch or delay, never introduce earlier.
- Reverse-curriculum spawns (starting partway through the maneuver) fix "learns
  the start, never the last mile".

## Training ops & reading a run

- wandb project `mjlab_microduck`; logs in `logs/<experiment_name>/`; resume with
  `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.
- **Warm start ≠ resume.** mjlab restores `common_step_counter`, so loading
  another task's checkpoint jumps every step-based curriculum to its final
  stage. Prefix `MICRODUCK_WARM_START=1` to restart counters at 0 while keeping
  weights/normalizer/optimizer. Only warm-start across compatible normalizers.
- Watch per iteration: mean reward rising AND episode length behaving as the
  task demands; every penalty term ≤ 0; the MAIN task term actually growing.
  `Episode_Reward/<term>` logs the WEIGHTED value.
- Budgets: episodic tricks ≈ 1000 iters at 4096 envs; gaits and
  curriculum-heavy recovery 4000–6000.
- **Measure before theorizing.** When a run "fails", run a headless eval of the
  checkpoint before changing rewards. Report what rollouts actually show ("rolls
  but face-plants 1 in 3"), not "it works!". The user decides when it's good
  enough.

## Sim2real footguns

- A fresh `uv sync` is the ground truth (HF Jobs and Colab run one). Keep
  `pyproject.toml` honest.
- **Wheels are per-architecture.** On linux-aarch64 PyPI's torch wheel is CPU-only
  and mjlab's `select_gpus()` indexes an empty list. `[tool.uv.sources]` routes
  torch to the cu129 index there; torch must stay a DIRECT dependency and the pin
  must stay `==` (locked by `tests/test_aarch64_cuda_torch.py`).
- Physics-aligned limits: a 25 cm robot tumbles at 3.5–5.5 rad/s naturally. Put
  anti-violence pressure on impacts and thrash (|a_z|, action_rate, support
  gates), not on rotation speed.
- IMU DR is zero-centered — it trains tolerance to misalignment magnitude and
  CANNOT compensate a systematic mounting bias (runtime calibration).
- Real deployments hot-swap ONNX policies with a shared obs contract — rehearse
  in `scripts/infer_policy.py` with the correct command-slot writes before
  touching the robot.
- **Unfiltered actions, baked normalizer, BAM actuators.** Breaking any of these
  looks fine in sim and fails on hardware.

## Docs

`docs/00-getting-started.md` … `docs/07-troubleshooting.md`. The Flamingo
walkthrough (`docs/03`) is the reference for adding a behavior; keep it in sync
with the code.
