# 05 — Train on a cloud GPU

A club laptop rarely has an NVIDIA GPU. Two supported paths:

1. **Hugging Face Jobs** — one flag on the normal `train` command. Recommended.
2. **Google Colab** — a notebook for people who prefer a UI; see
   [`notebooks/train_colab.ipynb`](../notebooks/train_colab.ipynb).

Both run the same code as a local run, from a fresh `uv sync`. If it works
locally only because of a hand-installed package, it will fail remotely.

## Hugging Face Jobs

### One-time setup

```bash
hf auth login          # or export HF_TOKEN
wandb login            # optional but strongly recommended
```

You need a Hugging Face account with Jobs credits. The namespace you run under
(personal or an org) also owns the source tarball, the checkpoint repo and the
job billing.

### Submit

Take any working local command and add `--hf-jobs`:

```bash
uv run train Mjlab-Flamingo-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 3000 --hf-jobs
```

Submission flags are consumed locally; everything else is forwarded to the job.

| flag | default | meaning |
|---|---|---|
| `--namespace <name>` | prompt / personal | account or org to run under |
| `--flavor l4x1` | `l4x1` | GPU flavor (`a10g-large`, `a100-large`, …) |
| `--timeout 12h` | `12h` | hard job limit |
| `--run-name <tag>` | `<task>-<timestamp>` | name of the HF model repo and run |
| `--detach` | off | submit and return; otherwise logs stream |
| `--dry-run` | off | build the tarball and print the job spec, submit nothing |
| `--no-wandb` | off | do not forward a wandb key (training then has no wandb) |
| `--uv-cache` | off | persistent uv cache bucket (usually slower than re-downloading) |

### What happens

1. `git ls-files` snapshots the repo (tracked + uncommitted, worktree-aware)
   into a tarball.
2. The tarball is uploaded to a private dataset
   `<namespace>/mjlab-microduck-src`.
3. A private model repo `<namespace>/<run-name>` is created for checkpoints.
4. A container runs `uv sync`, starts a background uploader that pushes
   `model_*.pt` every 60 s, then runs `uv run train <task> <your args>`.
5. wandb credentials are forwarded, so the run shows up live in
   `mjlab_microduck`.

If the job dies, the last uploaded checkpoint is on the Hub. Download it and
resume locally or in the next job:

```bash
uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.load-checkpoint model_1500.pt --agent.resume True
```

### First run checklist

- Run a 5-iteration smoke test **remotely** once (`--agent.max_iterations 5
  --env.scene.num-envs 64`). It validates the fresh `uv sync`, the dataset
  permissions and the uploader for a few cents.
- Check `Episode_Reward/<penalty> <= 0` in wandb within the first minutes.
- Watch `Episode_Reward/<main task term>`: total reward can rise on
  regularizers alone while the trick never happens.

See [scripts/hf/README.md](../scripts/hf/README.md) for the mechanism and known
failure modes (billing, token scopes, `403` on job listing).

## Google Colab

Open `notebooks/train_colab.ipynb` and run the cells:

1. check the GPU (`nvidia-smi`),
2. install `uv`,
3. clone the repo (or upload a zip),
4. `uv sync`,
5. smoke test,
6. a short training run.

Colab's free GPUs are slower than an `l4x1` and sessions are time-limited.
Use it to learn the loop and to run small experiments; use HF Jobs for a real
run. A free Colab session is enough for the Flamingo task at 256–1024 envs.

## Local NVIDIA box

If you have one, nothing special is required beyond `uv sync`. On
linux-aarch64 (DGX Spark / GB10, Jetson) PyPI's torch wheel is CPU-only; the
`pyproject.toml` routes torch to the cu129 index for that platform only. Do not
"fix" the direct `torch==` pin: it is what makes that routing apply. Locked by
`tests/test_aarch64_cuda_torch.py`.

## Budgets and run names

- Simple episodic tricks: ~1000 iterations at 4096 envs.
- Gaits and curriculum-heavy recovery: 4000–6000.
- Use `--agent.run-name` (`flamingo-widepose`, `flamingo-impact2`) so wandb
  runs are comparable. Change one thing per run.

## Warm starts

Loading another task's checkpoint is **not** the same as resuming. mjlab stores
`common_step_counter` in the checkpoint, so loading it jumps every step-based
curriculum to its final stage. Prefix `MICRODUCK_WARM_START=1` to restart the
counters at 0 while keeping weights, normalizer and optimizer:

```bash
MICRODUCK_WARM_START=1 uv run train Mjlab-Flamingo-Flat-MicroDuck \
    --agent.load-checkpoint stand_model.pt --env.scene.num-envs 4096
```

Only warm-start across tasks with compatible normalizers (do not seed a walking
env from a standing expert: its twist std is 0.005).
