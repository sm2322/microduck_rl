"""Task registration — the live registry behind `uv run list-envs`.

Four tasks, all flat terrain, all sharing the same 61-D observation contract
and the same 14-D action space, so the runtime can hot-swap their policies:

    Mjlab-Velocity-Flat-MicroDuck   walking (the base recipe)
    Mjlab-StandUp-Flat-MicroDuck    stand up after a fall
    Mjlab-SitStand-Flat-MicroDuck   commanded sit <-> stand
    Mjlab-Flamingo-Flat-MicroDuck   one-legged hold (the teaching example)

Adding a task: write ``microduck_<name>_env_cfg.py`` next to these, export a
``make_microduck_<name>_env_cfg(play=False)`` factory and a runner cfg, then
register it below. See docs/03-flamingo-walkthrough.md.
"""

from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in. A no-op without the flag. See docs/05-train-on-cloud.md.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    """mjlab runner with a small config-serialization fix.

    ``resolve_symmetry_config`` injects the env object into
    ``train_cfg["algorithm"]["symmetry_cfg"]`` in place. The MjSpec is not
    picklable, so we swap in a copy without ``_env`` for the yaml dump,
    leaving PPO's own reference untouched.
    """

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (  # noqa: E402
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from .microduck_standup_env_cfg import (  # noqa: E402
    MicroduckStandUpRlCfg,
    make_microduck_standup_env_cfg,
)
from .microduck_sitstand_env_cfg import (  # noqa: E402
    MicroduckSitStandRlCfg,
    make_microduck_sitstand_env_cfg,
)
from .microduck_flamingo_env_cfg import (  # noqa: E402
    MicroduckFlamingoRlCfg,
    make_microduck_flamingo_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Flamingo-Flat-MicroDuck",
    env_cfg=make_microduck_flamingo_env_cfg(),
    play_env_cfg=make_microduck_flamingo_env_cfg(play=True),
    rl_cfg=MicroduckFlamingoRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
