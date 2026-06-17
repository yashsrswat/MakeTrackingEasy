"""Convert Motion-X++ SMPL-X JSON clips into the NPZ format NMR's inference expects.

Motion-X++ stores motion as per-frame SMPL-X parameters in a camera coordinate
frame (X = person's left, Y = down, Z = depth into scene). NMR's inference loader
(`inference.load_smpl_data`) has two branches:

  - keys `trans/root_orient/pose_body`     -> assumed Z-up, code rotates Z-up->Y-up
  - keys `transl/global_orient/body_pose`  -> assumed Y-up, code applies NO rotation

Since Motion-X is neither Z-up nor Y-up, we apply our OWN rigid rotation `R` here
and emit the `transl/global_orient/body_pose` keys so NMR does not rotate again.
This keeps the full coordinate transform in one controllable place.

Default `R` = 180 deg about X = diag(1, -1, -1): flips the up-axis from Y-down to
Y-up. It is a proper rotation (det +1), so it does NOT mirror left/right.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# Named candidate transforms from Motion-X camera frame -> a world frame.
# All are proper rotations (det +1). Pick via --rot during the smoke test.
ROTATIONS = {
    "x180": np.diag([1.0, -1.0, -1.0]),   # Y-down -> Y-up (default)
    "z180": np.diag([-1.0, -1.0, 1.0]),   # 180 about Z
    "identity": np.eye(3),                 # no rotation (debug)
}


def load_motionx_clip(json_path):
    """Parse a Motion-X++ JSON into stacked SMPL-X arrays.

    Returns (root_orient (T,3), pose_body (T,63), trans (T,3), betas (10,)).
    """
    with open(json_path) as f:
        data = json.load(f)

    anns = data["annotations"]
    T = len(anns)
    root_orient = np.zeros((T, 3), dtype=np.float64)
    pose_body = np.zeros((T, 63), dtype=np.float64)
    trans = np.zeros((T, 3), dtype=np.float64)
    for i, ann in enumerate(anns):
        sp = ann["smplx_params"]
        root_orient[i] = sp["root_orient"]
        pose_body[i] = sp["pose_body"]
        trans[i] = sp["trans"]
    betas = np.asarray(anns[0]["smplx_params"]["betas"], dtype=np.float64)
    return root_orient, pose_body, trans, betas


def convert(json_path, out_path, rot="x180"):
    """Convert one Motion-X JSON to an NMR-ready NPZ. Returns the output path."""
    root_orient, pose_body, trans, betas = load_motionx_clip(json_path)
    R = ROTATIONS[rot]

    # Rotate translation: t' = R @ t
    transl = trans @ R.T

    # Rotate global orientation: R_world * R_root  (compose in rotation-matrix space)
    root_mat = Rotation.from_rotvec(root_orient).as_matrix()      # (T,3,3)
    new_root_mat = R[None] @ root_mat
    global_orient = Rotation.from_matrix(new_root_mat).as_rotvec()  # (T,3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        transl=transl.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        body_pose=pose_body.astype(np.float32),
        betas=betas.astype(np.float32),
    )
    return out_path, root_orient.shape[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Motion-X JSON -> NMR NPZ")
    ap.add_argument("--src", required=True, help="Motion-X .json file")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--rot", default="x180", choices=list(ROTATIONS),
                    help="Camera->world rotation (default x180: Y-down->Y-up)")
    args = ap.parse_args()
    out, T = convert(args.src, args.out, args.rot)
    print(f"{T} frames  rot={args.rot}  ->  {out}")
