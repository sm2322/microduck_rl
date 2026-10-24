"""Named joint poses for the Microduck, as plain Python dicts.

Deliberately dependency-free: importing this module does not pull in torch,
warp, mjlab or mujoco, so the same numbers can be used by

* the training env configs (``tasks/*_env_cfg.py``), and
* the CPU-only pose viewer (``scripts/view_pose.py``).

Angles are radians and joint names are the MJCF names. Only the two leg
chains are listed; neck/head default to the standing pose when a pose is
applied by the viewer.

Joint layout (14 servo joints, same on both robot models):
    0-4   left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8   neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13  right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)

When you change a pose, look at it before training on it:

    uv run scripts/view_pose.py --pose flamingo

then check the printed foot heights / tilt (or watch the viewer) and only
then launch a smoke test. A pose that is not a stable equilibrium — or that
is measured against the wrong robot revision — silently turns a reward into
an impossible target.
"""

# Trunk height (m above the floor) of the standing equilibrium, measured in
# sim on the ground-contact model. Re-measure if the robot model changes.
STAND_Z = 0.115

# Standing / home pose. Mirrors HOME_FRAME in robot/microduck_constants.py and
# the STAND keyframe in scene.xml. Leg pitch chain is leaned ~5 mm forward so
# the centre of mass sits over the ankle axis.
HOME = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

# Flamingo: stand on the LEFT leg, lift the RIGHT knee.
# Values chosen from forward kinematics on robot_groundcontact.xml: at this
# pose the right foot site is ~0.10 m above the floor (vs 0.003 m standing)
# and ~5 cm in front of the hip, so the lifted leg is clearly visible.
# The stance leg stays exactly at HOME.
FLAMINGO_LIFTED = {
    "right_hip_pitch": 1.05,   # HOME + 0.59 rad: thigh up/forward
    "right_knee": -1.50,       # heel tucked under the thigh
    "right_ankle": -0.4530,    # keep the foot pointing down (HOME value)
}

FLAMINGO = {**HOME, **FLAMINGO_LIFTED}

# Sit pose used by the SitStand task (see microduck_sitstand_env_cfg.py).
SIT = {
    **HOME,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}

POSES = {
    "home": HOME,
    "stand": HOME,
    "flamingo": FLAMINGO,
    "sit": SIT,
}


def overrides_for(joint_indices: dict[str, int], pose: dict[str, float]) -> dict[int, float]:
    """Translate a name->angle pose into the index->angle form used by mdp.

    ``joint_indices`` maps MJCF joint names to servo indices (the layout in the
    module docstring). Missing names are skipped, so the base pose is whatever
    the env's default joint position is.
    """
    return {joint_indices[name]: angle for name, angle in pose.items() if name in joint_indices}
