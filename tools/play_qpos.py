"""Interactive MuJoCo playback of an exported G1 qpos trajectory.

Plays a (T,36) qpos .npy (from mujoco_vis.py's `*_qpos.npy`) on the meshed G1.
Run LOCALLY on a machine with a display (not headless):

    python tools/play_qpos.py --traj smoke_out/STEP_OUT_clip1_qpos.npy

Controls: Space=pause, mouse=orbit/pan/zoom, ESC=quit.
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

# Reuse the same meshed scene (floor + light) as the headless renderer.
from mujoco_vis import load_scene  # noqa: E402  (run from tools/ or repo root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="(T,36) qpos .npy")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-loop", action="store_true")
    args = ap.parse_args()

    traj = np.load(args.traj)
    T = traj.shape[0]
    print(f"Loaded {T} frames; playback @ {args.fps} fps. Space=pause, ESC=quit.")

    model = load_scene()
    data = mujoco.MjData(model)
    frame, paused = 0, False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 130
        while viewer.is_running():
            t0 = time.perf_counter()
            data.qpos[:] = traj[frame]
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = [traj[frame, 0], traj[frame, 1], 0.8]
            viewer.sync()
            if not paused:
                frame += 1
                if frame >= T:
                    if args.no_loop:
                        break
                    frame = 0
            time.sleep(max(0, 1.0 / args.fps - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
