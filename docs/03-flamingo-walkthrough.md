# 03 — Flamingo walkthrough: build a behavior

Flamingo is the worked example. The task: **stand on the left foot and lift the
right knee**, like a flamingo. It is small enough to read in one sitting and
contains every ingredient a real trick needs: a target pose, a composite goal
reward, gates, a tighter fall termination, and a training run.

Files you will touch:

| file | what lives there |
|---|---|
| `src/mjlab_microduck/robot/poses.py` | the target pose (shared with the CPU viewer) |
| `src/mjlab_microduck/tasks/microduck_flamingo_env_cfg.py` | reward stack, terminations, commands, PPO cfg |
| `src/mjlab_microduck/tasks/mdp.py` | `flamingo_hold_score` — the composite reward |
| `tests/test_flamingo_cfg.py` | invariants that must stay true |
| `docs/03-flamingo-walkthrough.md` | this page |

## 1. Define the pose

Open `src/mjlab_microduck/robot/poses.py`. The stance leg is exactly the `HOME`
pose; only the lifted leg changes:

```python
FLAMINGO_LIFTED = {
    "right_hip_pitch": 1.05,   # thigh up/forward
    "right_knee": -1.50,       # heel tucked under the thigh
    "right_ankle": -0.4530,    # keep the foot pointing down
}
```

These numbers came from forward kinematics on `robot_groundcontact.xml`: the
right foot ends up ~0.10 m above the floor and ~5 cm in front of the hip. The
stance leg is untouched, so the trunk can stay at `STAND_Z`.

**Look at it before you train on it:**

```bash
uv run scripts/view_pose.py --pose flamingo
uv run scripts/view_pose.py --pose flamingo --settle 3 --no-viewer
```

The viewer shows the pose; `--settle 3` runs three seconds of physics and
reports trunk height, both foot heights and tilt. If it topples or the trunk
drops below 7 cm, the pose is not a stable target — change the pose, not the
reward. A 5 mm-wrong height once turned a reward into an impossible target for
days.

## 2. Read the env cfg

`microduck_flamingo_env_cfg.py` starts from the walking recipe:

```python
cfg = make_microduck_velocity_env_cfg(play=play)
```

That single line gives the robot, BAM actuators, domain randomization, obs
noise, delays, NaN guards and the 61-D command slots. The file then changes only
what defines the behavior.

### Robot and episode

```python
cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
cfg.episode_length_s = 8.0
```

The ground-contact model, so a failed hold is a real fall and not a body passing
through the floor.

### Stance-foot sensor

The stock contact sensor covers both feet. Lifting one foot would make it read
0.5 forever, so Flamingo adds a one-foot sensor:

```python
ContactSensorCfg(name="left_foot_stance", primary=... "left_foot_collision" ...)
```

The reward gate and the stance reward both use it.

### Reward stack

```python
W_HOLD = 4.0            # multiplicative goal score
W_POSE_BOOTSTRAP = 2.0  # wide Gaussian, discovery gradient
W_HEIGHT = 1.5
W_STANCE = 1.5
W_UPRIGHT = 2.0
W_ACTION_RATE = -0.1
W_BODY_ANG_VEL = -0.05
W_ANGULAR_MOMENTUM = -0.02
W_IMPACT = 0.05         # self-negating -> POSITIVE weight
W_SELF_COLLISION = -1.0
```

| term | function | role |
|---|---|---|
| `flamingo_hold` | `flamingo_hold_score` | **the goal**: pose × height × upright × stance contact |
| `flamingo_pose` | `pose_target_match` | wide additive bootstrap toward the leg pose |
| `flamingo_height` | `height_target_gaussian` | keep the trunk at standing height |
| `stance_foot_grounded` | `feet_grounded_reward` | keep the left foot down |
| `upright` | stock | don't lean |
| `trunk_impact` | `trunk_vertical_accel_penalty` | no hops / hard landings |
| `action_rate_l2`, `body_ang_vel`, `angular_momentum` | stock | jitter / thrash |
| `self_collisions` | stock | don't push the lifted leg into the body |

Why a product for the goal? An additive stack has a compromise basin: two feet
down, a slight lean, trunk a bit low — 80 % of every term. The product collapses
on any deficient factor. The Gaussians are deliberately **wide**
(`0.45 / 0.05 / 0.40`) so the untrained policy scores something and has a
gradient; tighten them only after the hold exists.

### Commands

Every command slot stays present (61-D contract), but the ranges are tiny:
`lin_vel_x` ±0.02, `ang_vel_z` ±0.05, head ±0.03. `body_command` is replaced by
`zero_command_padding` — the slot exists, the value is exactly zero.

### Terminations

```python
cfg.terminations["fell_over"].params["limit_angle"] = radians(55)
cfg.terminations["trunk_too_low"] = ... min_height=0.070 ...
cfg.terminations["nan_state"] = ...
```

The stock fall limit is 70°. A one-leg hold should end sooner, otherwise the
policy learns to catch itself with the lifted foot or the head. No reward is
paid on the terminal step, so falling is strictly worse than a bad hold.

### Physics

`nconmax=200`, solver iterations 30 / 50. The ground-contact model plus
one-leg loading produces more contacts than walking; the extra headroom avoids a
contact-overflow NaN on a failed hold.

## 3. Run the invariants

```bash
uv run --with pytest pytest tests/test_flamingo_cfg.py
```

The test locks:

- the pose only uses real joint names, and lifts the right leg only,
- the goal reward is positive and multiplicative,
- penalties keep the sign convention,
- the target pose matches `poses.py` (what the viewer shows),
- the stance sensor is left-foot-only,
- `body_command` is zero-padded to 6,
- the fall termination is 55° and the height floor exists.

When you invent your own task, copy this file and change the assertions. These
tests run on CPU and catch the mistakes that are expensive to find after a
4-hour run.

## 4. Smoke test, then train

```bash
# config errors for cents
uv run train Mjlab-Flamingo-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5

# the real run
uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096
```

Watch in wandb:

- `Episode_Reward/flamingo_hold` must become non-zero and grow. If it stays at
  zero, the Gaussians are too tight or the pose is unreachable.
- `Episode_Reward/trunk_impact` must be `<= 0`.
- episode length should grow toward 8 s / 0.02 = 400 steps as the hold
  consolidates, and drop when it is learning to catch itself.
- the video: is it on one foot, or is it hopping / leaning on the lifted foot?

Expected budget: a few hundred iterations for a recognizable hold, ~1000 for a
clean one at 4096 envs.

## 5. Tune it

| symptom | fix |
|---|---|
| never lifts the foot | widen `BOOTSTRAP_POSE_STD`, raise `W_POSE_BOOTSTRAP` |
| hops | raise `W_IMPACT`, lower `W_HOLD` for a while |
| leans on the lifted foot | the stance sensor gate is not wired / wrong foot |
| trunk sinks | tighten `HOLD_HEIGHT_STD`, raise `W_HEIGHT` |
| jitter | raise `W_ACTION_RATE` **after** the hold exists |
| falls constantly | widen `HOLD_UPRIGHT_STD`, or spawn closer to the target |

Change one number at a time and keep the run name, so the comparison is honest.

## 6. Variations to try

- **Right-foot flamingo:** swap `SUPPORT_FOOT`, `LIFTED_LEG_JOINTS`,
  `SUPPORT_SENSOR` and the pose overrides. Everything else stays.
- **Head turned while balancing:** give the `head_pose` command a real range and
  add a tracking reward. Keep the command slot; never delete it.
- **Freeze after 3 s:** add an episodic target rather than a held pose.
- **Balance on the toes:** change the stance leg's ankle in `poses.py`, then
  re-measure with the viewer.

For any of these: update the pose, look at it, update the test, smoke test,
train. See [02-reward-engineering.md](02-reward-engineering.md) for the rules
that keep the new behavior from being hacked.
