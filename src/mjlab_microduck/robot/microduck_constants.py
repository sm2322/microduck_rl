"""Robot configurations shared by every club task.

Two MJCF models are used:

* ``robot_walk.xml`` — walking model (feet-only contacts; falling is cheap).
* ``robot_groundcontact.xml`` — curated ground-contact model (soles, legs,
  trunk shells, head, jaw, battery, hips). Used by every task where the robot
  may touch the floor with more than its feet: stand-up, sit/stand, flamingo.

Both models have the same 14 servo joints in the same order, so observations
and actions are interchangeable and a policy trained on one model can be
played with the other.

Everything in this module is imported by the env cfgs; keep it free of task
logic. If you only want to *look* at the robot on a laptop, use
``scripts/view_pose.py`` (plain MuJoCo, no CUDA, no mjlab import).
"""

import os
from pathlib import Path

import mujoco
from mjlab_microduck.actuator import FrictionDRBamActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
MICRODUCK_GROUNDCONTACT_XML: Path = _ROBOT_DIR / "robot_groundcontact.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_GROUNDCONTACT_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


# The standing pose, in radians. Joint order matters for the head indexing
# helpers in ``tasks/mdp.py``:
#   0-4   left leg  (hip_yaw, hip_roll, hip_pitch, knee, ankle)
#   5-8   neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
#   9-13  right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
# Keep this pose in sync with the STAND keyframe in scene.xml / scene_walk.xml.
HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- BAM M6 actuator: the real XL330 model, not an ideal PD servo ------------
# Voltage control law + back-EMF + Coulomb/Stribeck/load-dependent friction.
# This is most of the sim2real gap on a 25 cm robot. See docs/01-the-robot.md.
#   target_names_expr excludes passive_* joints (the jaw linkage).
#   vin_range / vin_drop_gain_range model battery voltage and sag.
#   delay 3-6 control steps models the servo command latency.
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    delay_min_lag=3,
    delay_max_lag=6,
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    # `python -m mjlab_microduck.robot.microduck_constants` opens the training
    # model in the Warp viewer (needs an NVIDIA GPU). Laptop alternative:
    # `uv run scripts/view_pose.py --pose stand`.
    import mujoco.viewer as viewer

    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scene = Scene(SCENE_CFG, device=device)
    viewer.launch(scene.compile())
