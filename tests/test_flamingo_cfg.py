"""Config invariants for the Flamingo teaching task.

These run on CPU (no GPU, no training) and lock the things that are easy to
break while editing the worked example:

* the task still builds at all,
* it keeps the shared 61-D observation contract (command slots stay present),
* the goal reward is a POSITIVE multiplicative score and the penalties keep
  the negative-weight convention,
* the support-foot sensor and the tightened fall termination are wired,
* the target pose matches ``robot/poses.py`` (the pose the viewer shows).

Run: ``uv run --with pytest pytest tests/test_flamingo_cfg.py``
"""

import math

import torch

from mjlab_microduck.robot.poses import FLAMINGO, FLAMINGO_LIFTED, HOME, SIT, STAND_Z


def _cfg():
    from mjlab_microduck.tasks.microduck_flamingo_env_cfg import (
        make_microduck_flamingo_env_cfg,
    )

    return make_microduck_flamingo_env_cfg()


# ── Pose data ────────────────────────────────────────────────────────────────
def test_poses_only_override_named_joints():
    # Every pose key must be a real MJCF joint name used by the env cfgs; a typo
    # is silently ignored by overrides_for() and turns the reward into a
    # different pose than the one the viewer shows.
    for pose in (HOME, FLAMINGO, SIT):
        assert set(pose) <= set(HOME), f"unknown joint in pose: {set(pose) - set(HOME)}"
        for angle in pose.values():
            assert math.isfinite(angle)


def test_flamingo_lifts_the_right_leg_only():
    assert FLAMINGO["left_hip_pitch"] == HOME["left_hip_pitch"]
    assert FLAMINGO["left_knee"] == HOME["left_knee"]
    assert FLAMINGO["right_hip_pitch"] > HOME["right_hip_pitch"]
    assert FLAMINGO["right_knee"] < HOME["right_knee"]
    assert FLAMINGO_LIFTED["right_hip_pitch"] == FLAMINGO["right_hip_pitch"]


# ── Env cfg ──────────────────────────────────────────────────────────────────
def test_flamingo_cfg_builds():
    cfg = _cfg()
    assert cfg.episode_length_s == 8.0
    assert "flamingo_hold" in cfg.rewards


def test_goal_reward_is_positive_and_multiplicative():
    cfg = _cfg()
    term = cfg.rewards["flamingo_hold"]
    assert term.weight > 0, "flamingo_hold returns [0, 1]; a negative weight rewards failure"
    assert term.func.__name__ == "flamingo_hold_score"
    assert term.params["support_sensor"] == "left_foot_stance"


def test_penalties_keep_the_negative_weight_convention():
    cfg = _cfg()
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert cfg.rewards[name].weight <= 0, f"{name} is a cost and must be negative"
    # Self-negating microduck penalty -> POSITIVE weight.
    assert cfg.rewards["trunk_impact"].weight > 0
    assert cfg.rewards["trunk_impact"].func.__name__ == "trunk_vertical_accel_penalty"


def test_target_pose_matches_the_viewer_pose():
    cfg = _cfg()
    params = cfg.rewards["flamingo_hold"].params
    assert params["target_overrides"] == {11: FLAMINGO_LIFTED["right_hip_pitch"], 12: FLAMINGO_LIFTED["right_knee"]}
    assert params["target_height"] == STAND_Z


def test_stance_sensor_is_left_foot_only():
    cfg = _cfg()
    sensors = {s.name: s for s in cfg.scene.sensors}
    assert "left_foot_stance" in sensors
    sensor = sensors["left_foot_stance"]
    assert "left_foot_collision" in sensor.primary.pattern
    assert "right_foot" not in sensor.primary.pattern


def test_command_slots_stay_alive_but_near_zero():
    cfg = _cfg()
    twist = cfg.commands["twist"].ranges
    assert twist.lin_vel_x != (0.0, 0.0)
    assert twist.lin_vel_y != (0.0, 0.0)
    # Body command is zero-padded in obs (no consumer in this task), but the
    # slot must still exist so the 61-D contract is unchanged.
    for group in ("actor", "critic"):
        body = cfg.observations[group].terms["body_command"]
        assert body.func.__name__ == "zero_command_padding"
        assert body.params["dim"] == 6


def test_fall_termination_is_tighter_than_walking():
    cfg = _cfg()
    assert cfg.terminations["fell_over"].params["limit_angle"] == math.radians(55.0)
    assert "trunk_too_low" in cfg.terminations
    assert cfg.terminations["nan_state"].params.get("sensor_names"), (
        "flamingo nan_state no longer watches contact forces"
    )


# ── Reward helper ────────────────────────────────────────────────────────────
def test_feet_grounded_reward_normalises_single_foot_sensor():
    """A one-foot sensor must still reach 1.0 when that foot is down."""
    from mjlab_microduck.tasks import mdp as microduck_mdp

    class _Data:
        found = torch.ones(3, 1)

    class _Sensor:
        data = _Data()

    class _Scene:
        sensors = {"left_foot_stance": _Sensor()}

    class _Env:
        num_envs = 3
        device = "cpu"
        scene = _Scene()

    out = microduck_mdp.feet_grounded_reward(_Env(), "left_foot_stance")
    assert torch.allclose(out, torch.ones(3))
