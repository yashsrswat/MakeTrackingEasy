"""Render an NMR G1 retarget result on the real (meshed) G1 robot in MuJoCo.

NMR's own MuJoCo XML ships without mesh STLs, so we render on Bio2Bot's fully
meshed `assets/unitree_g1/g1.xml` instead. NMR's DOF vector is in an interleaved
order that differs from the robot's per-limb joint order, so we map DOF -> qpos
BY JOINT NAME (not position). NMR output is Z-up, matching MuJoCo, so no frame
conversion is needed.

Outputs an .mp4 (full clip) and a stills montage .png into smoke_out/.
Run with the gmr conda env's python; uses headless EGL.
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from inference import load_all, infer_single  # noqa: E402

G1_DIR = os.path.expanduser("~/dev/Bio2Bot/assets/unitree_g1")
G1_XML = os.path.join(G1_DIR, "g1.xml")

# NMR DOF (lab) index -> robot joint name. Order taken from the labelled comment
# in visualize.py ("body_mapping 索引含义"); "torso" == waist_pitch.
LAB_TO_JOINT = [
    "left_hip_pitch_joint",  "right_hip_pitch_joint",  "waist_yaw_joint",
    "left_hip_roll_joint",   "right_hip_roll_joint",   "waist_roll_joint",
    "left_hip_yaw_joint",    "right_hip_yaw_joint",    "waist_pitch_joint",
    "left_knee_joint",       "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_ankle_roll_joint",     "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_elbow_joint",          "right_elbow_joint",
    "left_wrist_roll_joint",     "right_wrist_roll_joint",
    "left_wrist_pitch_joint",    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",      "right_wrist_yaw_joint",
]

# Minimal scene: include the meshed robot (relative path so its meshdir resolves),
# add a floor + light. Written into G1_DIR as a temp file and loaded by path.
SCENE_XML = """
<mujoco model="nmr_vis">
  <include file="g1.xml"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="_floor" type="plane" size="20 20 0.05" rgba="0.8 0.85 0.9 1"/>
  </worldbody>
</mujoco>
"""


def load_scene():
    """Write the wrapper scene into G1_DIR (for path resolution), load, clean up."""
    tmp = os.path.join(G1_DIR, "_nmr_vis_tmp.xml")
    with open(tmp, "w") as f:
        f.write(SCENE_XML)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


def build_qpos(model, dof, trans, quat):
    """Assemble (T,36) qpos: free joint (pos+wxyz quat) + 29 hinges, mapped by name."""
    T = dof.shape[0]
    qpos = np.zeros((T, model.nq), dtype=np.float64)
    qpos[:, 0:3] = trans
    qpos[:, 3:7] = quat  # wxyz, matches MuJoCo free-joint convention
    for lab_i, jname in enumerate(LAB_TO_JOINT):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise KeyError(f"joint not found in model: {jname}")
        qpos[:, model.jnt_qposadr[jid]] = dof[:, lab_i]
    return qpos


def make_camera(lookat):
    cam = mujoco.MjvCamera()
    cam.distance = 3.2
    cam.elevation = -15
    cam.azimuth = 130
    cam.lookat[:] = lookat
    return cam


def render_frame(renderer, model, data, qpos_t, cam):
    data.qpos[:] = qpos_t
    mujoco.mj_forward(model, data)
    cam.lookat[0], cam.lookat[1] = qpos_t[0], qpos_t[1]  # track root in XY
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Converted .npz (transl/global_orient/body_pose)")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "smoke_out"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--stills-only", action="store_true", help="Skip mp4 (faster)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]

    print("Loading NMR model...")
    bundle = load_all()
    result, _ = infer_single(args.src, *bundle, apply_filter=not args.no_filter)
    if result is None:
        print("Sequence too short (<4 frames)."); return
    dof, trans, quat = result["dof"], result["root_trans"], result["root_rot_quat"]
    T = dof.shape[0]
    print(f"Inference OK: {T} frames")

    model = load_scene()
    data = mujoco.MjData(model)
    qpos = build_qpos(model, dof, trans, quat)

    # Export qpos (T,36) in g1.xml joint order for the interactive viewer
    # (Bio2Bot visualize.py playback, or tools/play_qpos.py).
    npy = os.path.join(args.out_dir, f"{stem}_qpos.npy")
    np.save(npy, qpos.astype(np.float32))
    print(f"Saved qpos -> {npy}  (shape {qpos.shape})")

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = make_camera([trans[0, 0], trans[0, 1], 0.8])

    # --- stills montage (6 frames) for quick eyeballing ---
    idxs = np.linspace(0, T - 1, 6).round().astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.4))
    for ax, t in zip(axes, idxs):
        ax.imshow(render_frame(renderer, model, data, qpos[t], cam))
        ax.set_title(f"frame {t}", fontsize=9); ax.axis("off")
    fig.suptitle(f"G1 MuJoCo: {stem}", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{stem}_mujoco.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    print(f"Saved stills montage -> {png}")

    if args.stills_only:
        return

    # --- full mp4 via ffmpeg pipe ---
    mp4 = os.path.join(args.out_dir, f"{stem}_mujoco.mp4")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for t in range(T):
        proc.stdin.write(render_frame(renderer, model, data, qpos[t], cam).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"Saved video ({T} frames @ {args.fps} fps) -> {mp4}")


if __name__ == "__main__":
    main()
