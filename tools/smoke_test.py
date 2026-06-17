"""Smoke-test NMR on a single converted clip and render a static PNG + HTML.

Pipeline:
  Motion-X NPZ (from motionx_to_npz.py)
    -> inference.infer_single          (SMPL-X -> G1 dof/root_trans/root_rot)
    -> visualize.compute_joint_positions (G1 dof -> 29 joint world positions, Z-up)
    -> matplotlib montage PNG  (so the result can be eyeballed headlessly)
    -> visualize.create_skeleton_animation -> interactive HTML

Run with the gmr conda env's python.
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from inference import load_all, infer_single                       # noqa: E402
from visualize import compute_joint_positions, create_skeleton_animation, BONES  # noqa: E402


def render_montage(joint_pos, out_png, n_frames=6, title=""):
    """Draw n evenly-spaced frames of the G1 skeleton as 3D subplots (Z up)."""
    T = joint_pos.shape[0]
    idxs = np.linspace(0, T - 1, min(n_frames, T)).round().astype(int)

    # Shared axis limits across frames so motion is comparable.
    p = joint_pos[idxs]
    mins = p.reshape(-1, 3).min(0) - 0.1
    maxs = p.reshape(-1, 3).max(0) + 0.1
    rng = (maxs - mins).max() / 2
    ctr = (maxs + mins) / 2

    fig = plt.figure(figsize=(3 * len(idxs), 3.6))
    for k, t in enumerate(idxs):
        ax = fig.add_subplot(1, len(idxs), k + 1, projection="3d")
        pos = joint_pos[t]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=8, c="tab:blue")
        for a, b in BONES:
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                    [pos[a, 2], pos[b, 2]], c="dimgray", lw=1.5)
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
        ax.set_title(f"frame {t}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (up)")
        ax.view_init(elev=12, azim=-70)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png, idxs.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Converted .npz (transl/global_orient/body_pose)")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "smoke_out"))
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]

    print("Loading NMR model (first run downloads checkpoint + SMPL-X)...")
    model, smplx_model, betas, smplx_mean, smplx_std, g1_mean, g1_std, device = load_all()
    print(f"Model loaded on {device}")

    result, timing = infer_single(
        args.src, model, smplx_model, betas,
        smplx_mean, smplx_std, g1_mean, g1_std,
        device, apply_filter=not args.no_filter,
    )
    if result is None:
        print("Sequence too short (<4 frames).")
        return

    dof, trans, quat = result["dof"], result["root_trans"], result["root_rot_quat"]
    T = dof.shape[0]
    print(f"Inference OK: {T} frames @ 30 FPS  ({timing['total']:.2f}s)")
    print(f"  root_trans z (up) range: {trans[:,2].min():.3f} .. {trans[:,2].max():.3f} m")
    print(f"  dof range: [{dof.min():.2f}, {dof.max():.2f}] rad")

    joint_pos = compute_joint_positions(dof, quat, trans)
    feet_z = joint_pos[:, :, 2].min()
    head_z = joint_pos[:, :, 2].max()
    print(f"  joint z extent: {feet_z:.3f} .. {head_z:.3f} m  (height ~{head_z-feet_z:.2f} m)")

    png = os.path.join(args.out_dir, f"{stem}_montage.png")
    render_montage(joint_pos, png, title=f"G1 retarget: {stem}")
    print(f"Saved montage PNG -> {png}")

    html = os.path.join(args.out_dir, f"{stem}.html")
    fig = create_skeleton_animation(dof, quat, trans)
    fig.write_html(html)
    print(f"Saved interactive HTML -> {html}")


if __name__ == "__main__":
    main()
