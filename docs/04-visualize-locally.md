# 04 — Visualize locally (CPU only)

You do not need a GPU to look at the robot or to watch a policy walk. Both tools
on this page use plain MuJoCo on the CPU. Training itself still needs a GPU.

## Look at a pose: `scripts/view_pose.py`

```bash
uv run scripts/view_pose.py --pose home
uv run scripts/view_pose.py --pose flamingo
uv run scripts/view_pose.py --pose sit --scene walk
uv run scripts/view_pose.py --pose flamingo --settle 3 --no-viewer
uv run scripts/view_pose.py --list
```

| flag | meaning |
|---|---|
| `--pose home\|stand\|flamingo\|sit` | named pose from `robot/poses.py` |
| `--scene groundcontact\|walk` | which robot model (default: ground-contact) |
| `--z <m>` | initial trunk height (default `STAND_Z`) |
| `--settle <s>` | run physics before measuring; catches toppling poses |
| `--no-viewer` | headless: print measurements and exit |

It prints, for the raw pose and after settling in parentheses:

```
trunk_z, left_foot_z, right_foot_z, tilt (degrees)
```

and warns when the settled pose has toppled (>30° tilt or trunk below 7 cm).
This is the check that saves days: a target pose that is not a stable
equilibrium turns a reward into an impossible objective.

Use it interactively while designing a pose: edit `robot/poses.py`, rerun,
watch the numbers. The viewer keeps physics running, so you can also see how the
pose collapses in real time.

## Watch a policy: `scripts/infer_policy.py`

This is the deployment rehearsal. It loads the exported ONNX, runs the policy at
50 Hz in MuJoCo, and simulates the servos with the **same BAM M6 XL330 model the
policy was trained against**. It is CPU-only and works on a laptop.

```bash
# a walking policy (61-D obs policies need --new-cmd-obs)
uv run scripts/infer_policy.py --walking walk.onnx --new-cmd-obs

# the Flamingo held pose (zero command)
uv run scripts/infer_policy.py --standing flamingo.onnx --new-cmd-obs

# sit <-> stand (the posture flag lives in twist[0]; press Y)
uv run scripts/infer_policy.py --sitstand sitstand.onnx --new-cmd-obs

# walk + stand, switching automatically on the velocity command
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx --new-cmd-obs
```

Keyboard controls (type in the terminal that launched the viewer — the window
does not capture keys):

| key | action |
|---|---|
| ↑ / ↓ | increase / decrease forward velocity |
| ← / → | strafe (walk robot) |
| A / E | turn left / right |
| SPACE | coast: zero all commands |
| T | pause / resume policy inference |
| P | random push (tests recovery) |
| Y | sit/stand toggle (`--sitstand`) |
| B / H | body-pose mode / head-pose mode |
| Q | quit |

Useful flags for sim2real work:

| flag | why |
|---|---|
| `--vin 7.4` | fix battery voltage instead of the training DR range |
| `--vin-drop-gain 0.1` | load-dependent voltage sag (0 disables) |
| `--kp-fw 200` | firmware P-gain, must match training |
| `--foot-friction 2.0` | emulate a grippier real PU sole (training used 0.7–1.3) |
| `--foot-solref 0.04` | soften foot contact to emulate sole compliance |
| `--delay 1 2` | actuator delay in timesteps |
| `--no-bam` | fall back to XML PD actuators (comparison only) |
| `--debug` | print observations and actions |
| `--record out.pkl` | save observations on Ctrl-C |
| `--save-csv out.csv` | log observations/actions |

### What to look for

- The policy should start from the **same pose the runtime starts it in**
  (standing). If it immediately collapses in sim, check the command you feed it:
  a policy trained with a zero command that receives stale head/body commands is
  out of distribution.
- `--no-bam` looking better than BAM is a red flag, not a result: the robot has
  BAM-like voltage-controlled servos, not ideal PD.
- If a policy walks here but falls on hardware, read
  [07-troubleshooting.md](07-troubleshooting.md) before changing the reward.

## Watch training checkpoints (needs a GPU)

`uv run play` renders with MuJoCo Warp and therefore uses the GPU:

```bash
uv run play Mjlab-Flamingo-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id>
```

It applies the observation normalizer itself, so it hides a hand-converted
checkpoint that forgot to bake the normalizer. Use it to inspect a run, but
always deploy ONNX produced by `scripts/export.py`.

## No display / headless

- `scripts/view_pose.py --no-viewer` prints numbers without opening a window.
- `scripts/infer_policy.py` needs the viewer; run it on a laptop or a desktop
  with X/Wayland, not on a headless server.
- On macOS, if the viewer window opens but you cannot type, click the terminal
  first: keys are read from stdin, not from the MuJoCo window.
