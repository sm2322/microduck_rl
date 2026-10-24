# 06 — Export and deploy to the robot

The runtime on the robot loads **ONNX**, one policy per file, and hot-swaps them
behind the shared 61-D observation contract. Two rules:

- Never hand-convert a checkpoint. The observation normalizer must be baked into
  the ONNX graph, or the policy sees unnormalized inputs at runtime. In-sim
  `play` hides this because it applies the normalizer itself.
- The graph shape is `[1, 61] -> [1, 14]`. A legacy 51-D policy is refused.

## 1. Export

```bash
uv run scripts/export.py Mjlab-Flamingo-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> \
    --checkpoint 3000 \
    --onnx-file flamingo.onnx
```

`--checkpoint` is the iteration number (`model_3000.pt`); omit it to use the
latest. Export runs the env once, so it needs the same GPU setup as training.
`scripts/export.py` is a thin wrapper around `mjlab_microduck.export`, the one
safe export path `publish` also uses.

## 2. Rehearse on CPU

Before touching hardware, run the exported file in CPU MuJoCo with the BAM
actuator model:

```bash
uv run scripts/infer_policy.py --standing flamingo.onnx --new-cmd-obs
```

- walking: `--walking walk.onnx --new-cmd-obs`
- sit/stand: `--sitstand sitstand.onnx --new-cmd-obs` (press `Y`)
- held pose: `--standing flamingo.onnx --new-cmd-obs`

Details and flags: [04-visualize-locally.md](04-visualize-locally.md).

## 3. What the runtime expects per task

| task | command it expects | behavior |
|---|---|---|
| Velocity | `twist = [vx, vy, wz]` | walks / turns; `0,0,0` = idle stand |
| StandUp | all zeros | recovers from a fall, then holds |
| SitStand | `twist[0]` is a flag: `0` = stand, `1` = sit | the same policy sits, holds and stands |
| Flamingo | all zeros | holds the one-leg pose |

The tricky one is the **flag**. Feeding all zeros to SitStand means "stand",
which looks exactly like "the policy ignores the button". Write the flag in
`twist[0]` in whatever client drives the policy. If you build a new behavior,
document its command convention in the model card (see `--twist-help` below).

## 4. Publish to the Hugging Face Hub

`uv run publish` creates a repo in the exact shape the robot's daemon loads:
`policy.onnx` + a schema-2 `manifest.json` + a README with install commands.

From a wandb run (the recommended path — export cannot be skipped):

```bash
uv run publish --task Mjlab-Flamingo-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <you>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 \
    --description "Stands on the left foot and lifts the right knee."
```

From an ONNX you already exported (validated, not re-exported):

```bash
uv run publish --onnx flamingo.onnx --repo <you>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 \
    --twist-help "[flag, side, 0]"
```

### Kinds

- **episodic** — runs for `--duration-s` and returns to standing by itself
  (a bow, a kick). Add `--chain` if a held button should repeat it.
- **perpetual gait** — `--slot walk` (or `stand`): loaded into a slot and runs
  until replaced.
- **perpetual held pose** — `--unwind-s <s>`: the daemon drives the idle twist
  for that long before handing back to the gait, so the robot is not released
  while on one foot. Flamingo is this kind.

`--kind` is required. `--dry-run` writes the repo contents to
`./publish-<name>/` without touching the Hub. Repos are private unless
`--no-private`. Existing `policy.onnx` is not overwritten without `--force`.

Before uploading, `publish`:

- checks the graph is `[1,61] -> [1,14]`,
- runs it on plausible inputs and refuses NaNs or a constant output,
- fills the `training` block from git + wandb (task, commit, branch, dirty
  flag, run, checkpoint).

Phase-driven or posture-flag behaviors (SitStand) are **not** publishable this
way; they are driven by the daemon and live in the official policy set.

### Walkthrough: publish a walking gait

```bash
uv run publish --task Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> \
    --repo <you>/microduck-my-walk --kind perpetual --slot walk

# on the robot
sudo robotctl policy load walk <you>/microduck-my-walk
```

## 5. Install on the robot

The repository README contains the exact commands, but the shape is:

```bash
# episodic skill
sudo robotctl policy add bow <you>/microduck-bow

# held pose (you choose how long)
sudo robotctl policy add flamingo <you>/microduck-flamingo --hold 5

# gait into a slot
sudo robotctl policy load walk <you>/microduck-my-walk

# run it
robotctl robot do flamingo
```

## 6. Before it leaves the bench

- Rehearse the same ONNX in `scripts/infer_policy.py`, from the pose the runtime
  starts it in, with the same command bytes the runtime will send.
- Check the battery voltage: a policy trained at 6.5–8.2 V should not be
  evaluated on a nearly empty pack in a way that surprises you.
- Log the run: the manifest `training` block points back to the exact wandb run
  and checkpoint. Keep that link when you share a remix (`--base-model`).
- If it behaves differently on hardware, do not re-tune the reward first. See
  [07-troubleshooting.md](07-troubleshooting.md).
