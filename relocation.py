import os
import sys
import json
import shutil
import sqlite3
import tempfile
import argparse
from collections import Counter

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import pycolmap

##############################################################################
# PRINT HELPERS
##############################################################################

def print_error(message):
  color = "\033[91m"
  reset = "\033[0m"
  print(f"{color}ERROR: {message}{reset}")

def print_success(message):
  color = "\033[92m"
  reset = "\033[0m"
  print(f"{color}SUCCESS: {message}{reset}")

def print_info(message):
  color = "\033[96m"
  reset = "\033[0m"
  print(f"{color}INFO: {message}{reset}")

def print_warning(message):
  color = "\033[93m"
  reset = "\033[0m"
  print(f"{color}WARNING: {message}{reset}")

def print_step(message):
  color = "\033[94m"
  reset = "\033[0m"
  print("\n")
  print(f"{color}{'='*100}{reset}")
  print(f"{color}RUNNING STEP: {message}{reset}")
  print(f"{color}{'='*100}{reset}\n")

##############################################################################
# ARGS
##############################################################################

def parse_args():
  parser = argparse.ArgumentParser(description="Visual camera relocalization using COLMAP SfM")
  parser.add_argument("--dataset", required=True, help="Path to dataset directory (e.g. datasets/home)")
  parser.add_argument("--image", required=True, help="Path to query image")
  parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold (default: 0.75)")
  return parser.parse_args()

##############################################################################
# LOAD RECONSTRUCTION
##############################################################################

def load_reconstruction(sfm_path):
  recon_path = os.path.join(sfm_path, "0")
  if not os.path.exists(recon_path):
    print_error(f"SfM reconstruction not found at {recon_path}")
    sys.exit(1)
  recon = pycolmap.Reconstruction(recon_path)
  print_info(f"Loaded reconstruction: {len(recon.cameras)} cameras, "
             f"{len(recon.images)} images, {len(recon.points3D)} 3D points")
  return recon

##############################################################################
# BUILD 3D DESCRIPTOR INDEX
##############################################################################

def build_3d_descriptor_index(recon, database_path):
  """
  Reads SIFT descriptors from the existing COLMAP SQLite database and builds
  a descriptor index keyed by 3D point. Uses the first track element per point
  to look up one representative descriptor.

  Returns:
    point3d_ids: np.ndarray (N,)      — 3D point IDs
    desc_index:  np.ndarray (N, 128) float32 — descriptor per 3D point
    xyz_array:   np.ndarray (N, 3)  float64  — world XYZ coordinates
  """
  conn = sqlite3.connect(database_path)
  c = conn.cursor()

  # Load all descriptors into memory keyed by image_id
  c.execute("SELECT image_id, rows, cols, data FROM descriptors")
  all_descriptors = {}
  for image_id, rows, cols, data in c.fetchall():
    all_descriptors[image_id] = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
  conn.close()

  point3d_ids = []
  desc_list = []
  xyz_list = []

  for pt3d_id, pt3d in recon.points3D.items():
    elem = pt3d.track.elements[0]
    image_id = elem.image_id
    point2d_idx = elem.point2D_idx

    if image_id not in all_descriptors:
      continue

    desc = all_descriptors[image_id][point2d_idx]  # (128,) uint8
    point3d_ids.append(pt3d_id)
    desc_list.append(desc)
    xyz_list.append(pt3d.xyz)

  point3d_ids = np.array(point3d_ids)
  desc_index = np.array(desc_list, dtype=np.float32)
  xyz_array = np.array(xyz_list, dtype=np.float64)

  print_info(f"Built descriptor index with {len(point3d_ids)} 3D points")
  return point3d_ids, desc_index, xyz_array

##############################################################################
# PREPARE QUERY IMAGE
##############################################################################

def prepare_query_image(image_path, tmp_dir, image_max_dimension):
  """Resize query image to image_max_dimension and save into tmp_dir."""
  with Image.open(image_path) as img:
    img = img.convert("RGB")
    width, height = img.size
    max_dim = max(width, height)
    if max_dim > image_max_dimension:
      scale = image_max_dimension / max_dim
      new_size = (int(width * scale), int(height * scale))
      img = img.resize(new_size, Image.Resampling.LANCZOS)
      print_info(f"Resized query image from ({width}, {height}) to {new_size}")
    else:
      print_info(f"Query image size ({width}, {height}) — no resize needed")
    out_name = os.path.basename(image_path)
    if not out_name.lower().endswith(('.jpg', '.jpeg', '.png')):
      out_name = out_name + '.jpg'
    out_path = os.path.join(tmp_dir, out_name)
    img.save(out_path)
  return out_path

##############################################################################
# EXTRACT QUERY FEATURES
##############################################################################

def extract_query_features(tmp_dir, tmp_db_path):
  """
  Extracts SIFT features from the (single) image in tmp_dir using pycolmap.

  Returns:
    keypoints:   np.ndarray (M, 6) float32 — [x, y, scale, ...]
    descriptors: np.ndarray (M, 128) uint8
  """
  if os.path.exists(tmp_db_path):
    os.remove(tmp_db_path)

  pycolmap.extract_features(
    tmp_db_path,
    tmp_dir,
    camera_mode=pycolmap.CameraMode.SINGLE,
  )

  conn = sqlite3.connect(tmp_db_path)
  c = conn.cursor()

  c.execute("SELECT image_id FROM images LIMIT 1")
  row = c.fetchone()
  if row is None:
    print_error("No image found in temporary database after feature extraction.")
    conn.close()
    sys.exit(1)
  image_id = row[0]

  c.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,))
  kp_row = c.fetchone()
  if kp_row is None or kp_row[2] is None:
    print_error("No keypoints extracted from query image.")
    conn.close()
    sys.exit(1)
  rows, cols, data = kp_row
  keypoints = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)

  c.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,))
  desc_row = c.fetchone()
  if desc_row is None or desc_row[2] is None:
    print_error("No descriptors extracted from query image.")
    conn.close()
    sys.exit(1)
  rows, cols, data = desc_row
  descriptors = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)

  conn.close()
  print_info(f"Extracted {len(keypoints)} keypoints from query image")
  return keypoints, descriptors

##############################################################################
# MATCH DESCRIPTORS
##############################################################################

def match_descriptors(query_descs, index_descs, ratio=0.75):
  """
  Nearest-neighbor matching with Lowe's ratio test.

  Returns:
    query_idxs: (K,) int — matched query keypoint indices
    db_idxs:    (K,) int — matched 3D point indices into index arrays
  """
  tree = cKDTree(index_descs)
  dists, idxs = tree.query(query_descs.astype(np.float32), k=2, workers=-1)

  ratio_mask = dists[:, 0] / (dists[:, 1] + 1e-8) < ratio
  query_idxs = np.where(ratio_mask)[0]
  db_idxs = idxs[ratio_mask, 0]

  print_info(f"Descriptor matching: {len(query_idxs)} matches after ratio test "
             f"(ratio={ratio}) from {len(query_descs)} query keypoints vs "
             f"{len(index_descs)} 3D points")
  return query_idxs, db_idxs

##############################################################################
# BUILD CAMERA FOR QUERY IMAGE
##############################################################################

def build_camera_for_query(image_path, recon):
  """
  Returns a pycolmap.Camera for the query image.
  Tries EXIF focal length first; falls back to scaling the reconstruction's
  most common camera to the query image dimensions.
  """
  # Strategy 1: EXIF
  try:
    cam = pycolmap.infer_camera_from_image(image_path)
    if cam is not None:
      print_info(f"Camera inferred from EXIF: {cam.model.name} "
                 f"{cam.width}x{cam.height} f={cam.params[0]:.1f}")
      return cam
  except Exception:
    pass

  # Strategy 2: scale most common reconstruction camera to query dims
  with Image.open(image_path) as img:
    q_w, q_h = img.size

  cam_sizes = Counter((c.width, c.height) for c in recon.cameras.values())
  ref_w, ref_h = cam_sizes.most_common(1)[0][0]
  ref_cam = next(c for c in recon.cameras.values() if c.width == ref_w and c.height == ref_h)

  scale = max(q_w, q_h) / max(ref_w, ref_h)
  focal = ref_cam.params[0] * scale

  cam = pycolmap.Camera(
    model="SIMPLE_RADIAL",
    width=q_w,
    height=q_h,
    params=[focal, q_w / 2.0, q_h / 2.0, 0.0],
  )
  print_info(f"Camera from reconstruction (scaled): SIMPLE_RADIAL "
             f"{cam.width}x{cam.height} f={focal:.1f}")
  return cam

##############################################################################
# RUN PNP
##############################################################################

def run_pnp(points2D, points3D, camera):
  """
  Runs PnP RANSAC via pycolmap.

  Returns result dict with 'cam_from_world', 'num_inliers', 'inlier_mask',
  or None if estimation failed.
  """
  if len(points2D) < 4:
    print_error(f"Too few 2D-3D correspondences: {len(points2D)} (need at least 4)")
    return None

  result = pycolmap.estimate_and_refine_absolute_pose(
    points2D.astype(np.float64),
    points3D.astype(np.float64),
    camera,
  )
  return result

##############################################################################
# PRINT POSE RESULT
##############################################################################

def print_pose_result(result):
  if result is None:
    print_error("Pose estimation failed — no result returned.")
    return

  cam_from_world = result["cam_from_world"]
  R = cam_from_world.rotation.matrix()  # 3×3, world→cam
  t = cam_from_world.translation        # (3,)

  # Camera center in world coordinates
  camera_center = -R.T @ t

  num_inliers = result.get("num_inliers", "?")
  inlier_mask = result.get("inlier_mask", [])

  print_success("Pose estimated successfully!")
  print_info(f"  Inliers: {num_inliers} / {len(inlier_mask)}")
  print_info(f"  Camera position in world:    [{camera_center[0]:.4f}, {camera_center[1]:.4f}, {camera_center[2]:.4f}]")
  print_info(f"  Translation vector t:        [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
  print_info(f"  Rotation matrix (world→cam):")
  for row in R:
    print_info(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")

##############################################################################
# MAIN
##############################################################################

def main():
  args = parse_args()

  dataset_path = args.dataset
  sfm_path = os.path.join(dataset_path, "sfm")
  database_path = os.path.join(dataset_path, "database.db")
  image_path = args.image

  for path, label in [(dataset_path, "dataset"), (database_path, "database"), (image_path, "query image")]:
    if not os.path.exists(path):
      print_error(f"{label} not found: {path}")
      sys.exit(1)

  config_path = os.path.join(dataset_path, "config.json")
  if not os.path.exists(config_path):
    print_error(f"config.json not found in {dataset_path}. Run pipeline.py first.")
    sys.exit(1)
  with open(config_path) as f:
    config = json.load(f)
  image_max_dimension = config["image_max_dimension"]
  print_info(f"Loaded config: image_max_dimension={image_max_dimension}")

  print_step("Load SfM Reconstruction")
  recon = load_reconstruction(sfm_path)

  print_step("Build 3D Descriptor Index")
  _, desc_index, xyz_array = build_3d_descriptor_index(recon, database_path)

  tmp_dir = tempfile.mkdtemp(prefix="reloc_")
  try:
    print_step("Prepare Query Image")
    resized_path = prepare_query_image(image_path, tmp_dir, image_max_dimension)
    tmp_db_path = os.path.join(tmp_dir, "query.db")

    print_step("Extract Query Features")
    query_kp, query_desc = extract_query_features(tmp_dir, tmp_db_path)

    print_step("Match Descriptors")
    q_idxs, db_idxs = match_descriptors(query_desc.astype(np.float32), desc_index, ratio=args.ratio)

    if len(q_idxs) < 4:
      print_error(f"Insufficient matches after ratio test: {len(q_idxs)}. "
                  "Try lowering --ratio or using a query image from the same scene.")
      sys.exit(1)

    points2D = query_kp[q_idxs, 0:2].astype(np.float64)   # (K, 2) x,y
    points3D = xyz_array[db_idxs].astype(np.float64)       # (K, 3)

    print_step("Build Query Camera Model")
    camera = build_camera_for_query(resized_path, recon)

    print_step("Run PnP Pose Estimation")
    result = run_pnp(points2D, points3D, camera)

    print_step("Results")
    print_pose_result(result)

  finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
