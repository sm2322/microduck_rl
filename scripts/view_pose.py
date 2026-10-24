#!/usr/bin/env python3
"""Look at a named Microduck pose in plain MuJoCo — no GPU, no mjlab, no warp.

This is the "does the pose even make sense?" tool. It loads the MJCF directly
with ``mujoco``, drops the robot at the pose you asked for, optionally lets the
physics settle for a second, and prints the numbers a reward function will
eventually be graded against:

    trunk height, left/right foot height, trunk tilt

Use it before training anything that targets a pose (Flamingo, SitStand, a new
trick). A pose that is not a stable equilibrium, or one that is 5 mm off the
real robot revision, turns the reward into an impossible target and you find
out days later. See docs/04-visualize-locally.md.

Examples
--------
    uv run scripts/view_pose.py --pose flamingo
    uv run scripts/view_pose.py --pose sit --scene walk --no-viewer
    uv run scripts/view_pose.py --pose home --settle 3

Dependencies: only ``mujoco`` (installed with the project). Intentionally does
NOT import ``mjlab``/``torch``, so it works on any laptop.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab_microduck.robot.poses import POSES, STAND_Z  # noqa: E402

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck"
SCENES = {
    "groundcontact": ROBOT_DIR / "scene.xml",
    "walk": ROBOT_DIR / "scene_walk.xml",
}

TRUNK_BODY = "trunk_base"
FOOT_SITES = ("left_foot", "right_foot")


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float], z: float) -> None:
    """Write a name->angle pose into qpos (and matching ctrl) by joint name."""
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = (0.0, 0.0, z)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)

    for name, angle in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  ! joint {name!r} not in this model; skipped")
            continue
        data.qpos[model.jnt_qposadr[jid]] = angle

    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    mujoco.mj_forward(model, data)


def measure(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Trunk height/tilt and foot heights from the current state."""
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
    trunk_z = float(data.xpos[body, 2])
    # Tilt = angle between the trunk's local +z and world +z.
    up = data.xmat[body].reshape(3, 3)[:, 2]
    tilt_deg = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))

    out = {"trunk_z": trunk_z, "tilt_deg": tilt_deg}
    for site in FOOT_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        out[site] = float(data.site_xpos[sid, 2]) if sid >= 0 else float("nan")
    return out


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float, timestep: float) -> None:
    """Hold ctrl at the pose and step physics; catches poses that topple."""
    n = max(1, int(seconds / timestep))
    for _ in range(n):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Passive MuJoCo viewer; physics keeps running so you can see stability."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose", default="home", choices=sorted(POSES), help="named pose to show")
    parser.add_argument(
        "--scene",
        default="groundcontact",
        choices=sorted(SCENES),
        help="robot model: groundcontact (soles/legs collide) or walk (feet only)",
    )
    parser.add_argument("--z", type=float, default=STAND_Z, help=f"initial trunk height (default {STAND_Z})")
    parser.add_argument(
        "--settle",
        type=float,
        default=0.0,
        help="seconds of physics to run before measuring (0 = measure the raw pose)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="headless: measure and exit")
    parser.add_argument("--list", action="store_true", help="list available poses and exit")
    args = parser.parse_args()

    if args.list:
        print("poses:", ", ".join(sorted(POSES)))
        return 0

    scene = SCENES[args.scene]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    pose = POSES[args.pose]
    apply_pose(model, data, pose, args.z)
    before = measure(model, data)

    print(f"scene : {scene.name}")
    print(f"pose  : {args.pose} ({len(pose)} joints set), initial trunk z = {args.z:.3f} m")
    print(
        f"  raw    trunk_z={before['trunk_z']:.4f}  "
        f"left_foot_z={before['left_foot']:.4f}  right_foot_z={before['right_foot']:.4f}  "
        f"tilt={before['tilt_deg']:.1f} deg"
    )

    if args.settle > 0:
        settle(model, data, args.settle, dt)
        after = measure(model, data)
        print(
            f"  settled({args.settle:.1f}s) trunk_z={after['trunk_z']:.4f}  "
            f"left_foot_z={after['left_foot']:.4f}  right_foot_z={after['right_foot']:.4f}  "
            f"tilt={after['tilt_deg']:.1f} deg"
        )
        if after["tilt_deg"] > 30.0 or after["trunk_z"] < 0.070:
            print("  ! pose toppled: not a stable equilibrium (tilt>30 deg or trunk<0.070 m)")

    if args.no_viewer:
        return 0

    try:
        run_viewer(model, data)
    except Exception as exc:  # headless machines / no display
        print(f"viewer unavailable ({exc}); rerun with --no-viewer", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
