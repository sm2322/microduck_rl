# 01 — The robot and the simulation contract

## The physical robot

- ~25 cm tall, ~800 g, two legs, 14 Dynamixel **XL330** servos.
- Trained and deployed at **50 Hz** (20 ms control period).
- No wheel, no arm: balance is the whole game.
- The head is heavy relative to the body (~38 % of the mass): it visibly
  oscillates while walking. Do not punish that oscillation; see
  [02-reward-engineering.md](02-reward-engineering.md).

## Joint layout (memorize this)

Fourteen actuated joints, always in this order:

| index | joint | side |
|---|---|---|
| 0 | `hip_yaw` | left |
| 1 | `hip_roll` | left |
| 2 | `hip_pitch` | left |
| 3 | `knee` | left |
| 4 | `ankle` | left |
| 5 | `neck_pitch` | neck |
| 6 | `head_pitch` | head |
| 7 | `head_yaw` | head |
| 8 | `head_roll` | head |
| 9 | `hip_yaw` | right |
| 10 | `hip_roll` | right |
| 11 | `hip_pitch` | right |
| 12 | `knee` | right |
| 13 | `ankle` | right |

Left leg first, then the four neck/head joints, then the right leg. The action
vector and the ctrl vector use this order. In `mdp.py`, prefer the helpers
`_servo_joint_ids` / `_servo_joint_pos` over hard-coded indices anyway — they
survive model revisions.

The standing pose (`HOME`) is in `src/mjlab_microduck/robot/poses.py` and
mirrors the `STAND` keyframe in the MJCF scenes. Height of the trunk at that
equilibrium: `STAND_Z = 0.115 m`.

## The observation contract (61 numbers)

Every policy in the family, club or official, consumes the same 61-D actor
observation:

```
48  proprioception
    joint positions (14), joint velocities (14), last action (14),
    base angular velocity (3), projected gravity (3)
13  command block
    twist (3)         [vx, vy, wz]
    head_pose (4)     [neck_pitch, head_pitch, head_yaw, head_roll] deltas
    body_pose (6)     [x, y, z, roll, pitch, yaw] deltas
```

Two rules follow from this, and breaking either breaks deployment:

1. **Never delete a command slot.** An env that does not use a slot keeps the
   observation term and feeds a constant (usually zeros, sometimes a tiny
   sampling range so the input neurons stay alive).
2. **Never reorder it.** The runtime writes this buffer; the policy does not
   know which task trained it.

A slot may carry a *flag* instead of a velocity. SitStand puts the posture flag
in `twist[0]` (0 = stand, 1 = sit); Flamingo uses all-zero commands. This is why
"press the button and nothing happens" usually means the runtime fed zeros to a
policy that expected a flag — see [07-troubleshooting.md](07-troubleshooting.md).

## Actuators: BAM, not ideal PD

The policies train and deploy against the [BAM](https://github.com/Rhoban/bam)
M6 model of the XL330:

- voltage control law with back-EMF,
- Coulomb / Stribeck / load-dependent friction,
- firmware position loop (`kp ≈ 200`) outside the simulated motor,
- randomized battery voltage (6.5–8.2 V), voltage sag under load, command
  delay and friction magnitude, per environment.

At this scale the servo is most of the sim2real gap, so replacing BAM with an
ideal PD actuator in training guarantees a policy that behaves differently on
hardware. `scripts/infer_policy.py` uses the same model on CPU
(`bam.mujoco.MujocoController`); `--no-bam` exists only for comparison.

Consequence for reward code: `dof_frictionloss` is zeroed under BAM, so joint
friction domain randomization must scale the actuator's `friction_scale`.
Randomizing `dof_frictionloss` is a silent no-op.

## Robot models

MJCF models are exported from Onshape with
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot). Two survive in
the club edition, both with the same 14 joints in the same order:

| model | collisions | used by |
|---|---|---|
| `robot_walk.xml` | feet only | Velocity |
| `robot_groundcontact.xml` | feet, legs, trunk shells, head, jaw, battery, hips | StandUp, SitStand, Flamingo |

`scene.xml` wraps the ground-contact model with a floor, lighting and
`STAND`/`SIT`/`FOLD` keyframes; `scene_walk.xml` does the same for the walking
model. `scripts/view_pose.py` loads these directly.

The walking model makes falling cheap (only the feet collide) — good for
learning a gait. Any task where the robot is allowed to touch the floor with its
body uses the ground-contact model, so a failure looks and costs like a real
fall.

## Domain randomization and noise

The walking recipe (`make_microduck_velocity_env_cfg`) wires the full stack, and
the other tasks inherit it:

- foot friction, trunk mass, centre of mass, motor strength, battery voltage,
  voltage sag, command delay, encoder bias, IMU mounting misalignment,
- observation noise on joint positions/velocities and IMU,
- a NaN guard on the critic observation terms and a `nan_state` termination
  that also watches contact forces.

`ENABLE_*` toggles at the top of `microduck_velocity_env_cfg.py` switch groups
on and off. Custom DR functions must restore the compile-time default before
applying a new sample; accumulating randomizers degrade every long run and are
hard to spot (see AGENTS.md).

## What "transfer" means here

A policy transfers when it survives the differences the simulation does *not*
model: a systematic IMU mounting bias, a slightly different floor, a battery
that sags, a servo with more play than the model. The training recipe covers
magnitudes, not systematic offsets — those are calibrated at runtime. Do not
try to fix a mounting bias with more randomization; fix it in calibration.
