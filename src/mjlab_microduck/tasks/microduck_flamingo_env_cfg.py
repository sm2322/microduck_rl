"""Microduck Flamingo — the club teaching task (one-legged balance).

Goal: stand on the LEFT foot and lift the RIGHT knee, like a flamingo.

This is the worked example for "Behavior Architect" work: a small, readable
env cfg that reuses the walking recipe's robot, domain randomization, obs
noise and delays (via ``make_microduck_velocity_env_cfg``) and replaces only
the things that define the behavior:

* the robot model (ground-contact model, so falls look right),
* the reward stack (hold the pose, don't farm partial credit, don't fall),
* the command slots (all commands are ~zero — the behavior is a held pose),
* the terminations (tilt or drop the trunk → episode over).

Read ``docs/03-flamingo-walkthrough.md`` while editing this file. Every
number you would want to change is a named constant at the top.

The observation layout stays the shared 61-D contract:
48 proprioception + [twist(3), head_pose(4), body_pose(6)]. Never delete a
command slot; zero-pad it instead.

Training:

    uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 64 \
        --agent.max_iterations 5          # smoke test first
    uv run train Mjlab-Flamingo-Flat-MicroDuck --env.scene.num-envs 4096
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.robot.poses import FLAMINGO_LIFTED, STAND_Z
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    ENABLE_ENCODER_BIAS,
    ENABLE_IMU_ORIENTATION_RANDOMIZATION,
    ENCODER_BIAS_RANGE,
    IMU_ORIENTATION_RANDOMIZATION_ANGLE,
    make_microduck_velocity_env_cfg,
)

# ── What to tune first ───────────────────────────────────────────────────────
# Support foot: "left" means stand on the left leg and lift the right knee.
SUPPORT_FOOT = "left"
SUPPORT_SENSOR = "left_foot_stance"

# Which leg to lift (servo joint indices; see the layout in mdp.py).
LIFTED_LEG_JOINTS = [9, 10, 11, 12, 13]

# Servo joint indices of the whole leg chain. The pose reward covers BOTH
# legs: the stance leg is held at its HOME default, the lifted leg at the
# target overrides below.
LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Target overrides for the lifted leg: {servo index: radians}.
# Index 11 = right hip_pitch, 12 = right knee. Values live in
# robot/poses.py so the CPU viewer (scripts/view_pose.py) shows the same pose.
FLAMINGO_TARGET_OVERRIDES = {
    11: FLAMINGO_LIFTED["right_hip_pitch"],
    12: FLAMINGO_LIFTED["right_knee"],
}

# Held-pose target. Trunk height is measured in sim (robot/poses.py): it is the
# height of the standing equilibrium, and the flamingo keeps the same trunk
# height while one leg is tucked.
TARGET_HEIGHT = STAND_Z

# Reward weights. Task terms are positive (they return [0, 1]); penalties use
# the stock sign convention (cost >= 0 → negative weight).
W_HOLD = 4.0          # multiplicative goal score (pose × height × upright × support)
W_POSE_BOOTSTRAP = 2.0  # wide Gaussian on leg joints — discovery gradient
W_HEIGHT = 1.5
W_STANCE = 1.5        # keep the support foot on the floor
W_UPRIGHT = 2.0       # already the walking recipe's value
W_ACTION_RATE = -0.1  # jitter
W_BODY_ANG_VEL = -0.05
W_ANGULAR_MOMENTUM = -0.02
W_IMPACT = 0.05       # self-negating |a_z| penalty -> POSITIVE weight
W_SELF_COLLISION = -1.0

# Pose Gaussians. Start WIDE (0.45 / 0.05 / 0.40) so the freshly-initialised
# policy scores something everywhere; tighten only after the behavior exists.
HOLD_POSE_STD = 0.45
HOLD_HEIGHT_STD = 0.05
HOLD_UPRIGHT_STD = 0.40   # ≈ 23°; a fallen robot scores ~0
BOOTSTRAP_POSE_STD = 0.80
BOOTSTRAP_HEIGHT_STD = 0.06

# Episode / termination.
EPISODE_LENGTH_S = 8.0
FALL_HEIGHT = 0.070   # trunk below this = on the floor (standing is 0.115)
FALL_TILT_DEG = 55.0
# The stock velocity termination uses 70°; a one-leg hold should end sooner so
# the policy does not learn to catch itself with the lifted foot or the head.
FALL_TILT_DEG_STOCK = 70.0


def make_microduck_flamingo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Flamingo one-leg-hold environment."""

    # Start from the walking recipe: robot config, BAM actuators, domain
    # randomization, obs noise/delays, NaN guard, command obs slots — all free.
    cfg = make_microduck_velocity_env_cfg(play=play)

    # ── Robot model ──────────────────────────────────────────────────────────
    # Ground-contact model: the body can physically hit the floor, so a failed
    # hold ends in a believable fall instead of a body passing through it.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Support-foot contact sensor ──────────────────────────────────────────
    # One sensor on the stance foot only. The stock two-foot sensor cannot be
    # used as a gate (lifting one foot would read 0.5 forever).
    support_geom = f"{SUPPORT_FOOT}_foot_collision"
    stance_contact_cfg = ContactSensorCfg(
        name=SUPPORT_SENSOR,
        primary=ContactMatch(mode="geom", pattern=rf"^{support_geom}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=False,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, stance_contact_cfg)

    # ── Rewards: drop the walking terms ──────────────────────────────────────
    # Everything below exists to make a gait track a velocity command. None of
    # it defines "hold a pose on one leg", so it goes.
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    # ── Rewards: the flamingo stack ──────────────────────────────────────────
    # The goal state as ONE multiplicative score. Partial poses (two feet,
    # leaning, hopping, trunk low) collapse to ~0, so the policy cannot farm
    # 80 % of every separate term.
    cfg.rewards["flamingo_hold"] = RewardTermCfg(
        func=microduck_mdp.flamingo_hold_score,
        weight=W_HOLD,
        params={
            "target_overrides": FLAMINGO_TARGET_OVERRIDES,
            "joint_indices": LEG_JOINTS,
            "pose_std": HOLD_POSE_STD,
            "target_height": TARGET_HEIGHT,
            "height_std": HOLD_HEIGHT_STD,
            "upright_std": HOLD_UPRIGHT_STD,
            "support_sensor": SUPPORT_SENSOR,
        },
    )

    # Wide, additive shaping so the policy gets a gradient toward the pose
    # BEFORE it accidentally completes the hold. This is the "bootstrap" term:
    # once the hold score is clearly non-zero in wandb, this can be reduced.
    cfg.rewards["flamingo_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=W_POSE_BOOTSTRAP,
        params={
            "target_overrides": FLAMINGO_TARGET_OVERRIDES,
            "joint_indices": LEG_JOINTS,
            "std": BOOTSTRAP_POSE_STD,
        },
    )
    cfg.rewards["flamingo_height"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=W_HEIGHT,
        params={
            "target_height": TARGET_HEIGHT,
            "std": BOOTSTRAP_HEIGHT_STD,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Keep the support foot down (positive reward, 1.0 while it is in contact).
    cfg.rewards["stance_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=W_STANCE,
        params={"sensor_name": SUPPORT_SENSOR},
    )

    # Always-on upright floor (stock walking term, same weight as walking).
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = W_UPRIGHT

    # A hard landing / hop pays on every step of the impact, not just once.
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=W_IMPACT,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ── Regularisers (match the walking recipe) ──────────────────────────────
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL
    cfg.rewards["angular_momentum"].weight = W_ANGULAR_MOMENTUM
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=W_SELF_COLLISION,
        params={"sensor_name": "self_collision"},
    )

    # ── Commands: keep every slot alive, but ask for ~nothing ────────────────
    # The held pose is the behavior; commands stay near zero. Tiny non-zero
    # ranges keep the command input neurons alive for later curricula, and the
    # obs layout stays the shared 61-D contract.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.5
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (-0.02, 0.02)
    twist.ranges.lin_vel_y = (-0.02, 0.02)
    twist.ranges.ang_vel_z = (-0.05, 0.05)

    head = cfg.commands["head_pose"]
    head.ranges = (
        (-0.03, 0.03),   # neck_pitch
        (-0.03, 0.03),   # head_pitch
        (-0.03, 0.03),   # head_yaw
        (-0.01, 0.01),   # head_roll
    )

    body = cfg.commands["body_pose"]
    body.ranges = tuple((lo * 0.2, hi * 0.2) for lo, hi in body.ranges)

    # ── Curricula: keep DR ramps, drop task-specific schedules ───────────────
    for name in (
        "action_rate_weight",
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(name, None)

    # ── Terminations ─────────────────────────────────────────────────────────
    # "fell_over" exists in the base recipe at 70°; tighten it and add a
    # trunk-height floor. No reward is paid on the terminal step, so falling
    # is strictly worse than a bad hold.
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(FALL_TILT_DEG)
    cfg.terminations["trunk_too_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={
            "min_height": FALL_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # ── Resets / physics ─────────────────────────────────────────────────────
    # Spawn standing just above the equilibrium: the hold is learned from a
    # two-foot start, which is the deployment situation (the robot is standing
    # when the trick is triggered).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Ground-contact model + one-leg loading puts more contact points in play
    # than walking; give the solver room so a failed hold does not produce a
    # contact-overflow NaN (the SitStand recipe's values).
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # Keep the IMU/encoder-bias DR exactly as the walking recipe wires it —
    # this is what makes the policy transferable to the real robot.
    assert ENABLE_ENCODER_BIAS and ENABLE_IMU_ORIENTATION_RANDOMIZATION
    cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
    cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
    cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    for term, func in (
        ("base_ang_vel", microduck_mdp.base_ang_vel_imu_misaligned),
        ("projected_gravity", microduck_mdp.projected_gravity_imu_misaligned),
    ):
        cfg.observations["actor"].terms[term].func = func
        cfg.observations["actor"].terms[term].params = {
            "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE
        }

    # Body pose command has no consumer in this task; keep the obs term but
    # make it exact: zero padding, so the slot never carries noise.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    return cfg


# ── PPO runner ───────────────────────────────────────────────────────────────
# Same network / PPO hyperparameters as walking, different experiment name so
# logs and checkpoints stay separate.
MicroduckFlamingoRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="flamingo",
    run_name="flamingo",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)
