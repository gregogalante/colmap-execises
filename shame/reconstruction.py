"""
reconstruction.py
=================
Dense 3D reconstruction from SfM output (pipeline.py).
100% CPU-only — no CUDA, no NVIDIA GPU required.

Pipeline:
  1. Load SfM cameras + poses from pycolmap
  2. For each image, find K nearest neighbours by camera centre
  3. Stereo-rectify each pair (OpenCV) + run StereoSGBM
  4. Back-project depth maps to coloured 3D points
  5. Merge all point clouds, remove outliers (Open3D)
  6. Poisson surface reconstruction (Open3D)
  7. Save fused.ply + meshed-poisson.ply

Dependencies: pycolmap, opencv-python, open3d, numpy, scipy, Pillow
"""

import os
import time
import shutil

import numpy as np
import cv2
import open3d as o3d
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree

##############################################################################
# CONFIG
##############################################################################

DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
DATASET_PATH = os.path.join(DATASETS_PATH, 'banana')   # same dataset as pipeline.py
DATASET_RESET = False   # True → delete dense/ and redo everything

IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
SFM_PATH = os.path.join(DATASET_PATH, 'sfm')
SFM_RECONSTRUCTION_PATH = os.path.join(SFM_PATH, '0')
DENSE_PATH = os.path.join(DATASET_PATH, 'dense')

# How many nearest-neighbour views to use per reference image.
# More → denser result but much slower.
K_NEIGHBOURS = 5

# StereoSGBM parameters (tuned for quality on CPU).
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 128   # must be divisible by 16
SGBM_BLOCK_SIZE = 7
SGBM_P1 = 8 * 3 * SGBM_BLOCK_SIZE ** 2
SGBM_P2 = 32 * 3 * SGBM_BLOCK_SIZE ** 2

# Depth filtering: discard points farther than this from the sparse cloud centroid.
MAX_DEPTH = 50.0        # metres (scene-dependent)
MIN_DEPTH = 0.05

# Open3D outlier removal.
NB_NEIGHBORS = 30
STD_RATIO = 2.0

# Poisson reconstruction depth (higher → more detail, slower).
POISSON_DEPTH = 9

##############################################################################
# PRINT HELPERS
##############################################################################

def print_error(message):
    print(f"\033[91mERROR: {message}\033[0m")

def print_success(message):
    print(f"\033[92mSUCCESS: {message}\033[0m")

def print_info(message):
    print(f"\033[96mINFO: {message}\033[0m")

def print_warning(message):
    print(f"\033[93mWARNING: {message}\033[0m")

def print_step(message):
    print("\n")
    print(f"\033[94m{'='*100}\033[0m")
    print(f"\033[94mRUNNING STEP: {message}\033[0m")
    print(f"\033[94m{'='*100}\033[0m\n")

##############################################################################
# CAMERA MODEL HELPERS
##############################################################################

def camera_to_opencv(camera):
    """
    Convert a pycolmap Camera to an OpenCV K matrix and distortion vector.
    Returns (K [3x3], dist [4,], width, height).
    """
    p = camera.params
    w, h = camera.width, camera.height
    model = camera.model.name  # e.g. 'SIMPLE_PINHOLE'

    if model == 'SIMPLE_PINHOLE':
        # params: f, cx, cy
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4)

    elif model == 'PINHOLE':
        # params: fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4)

    elif model == 'SIMPLE_RADIAL':
        # params: f, cx, cy, k1
        f, cx, cy, k1 = p[0], p[1], p[2], p[3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0, 0, 0], dtype=np.float64)

    elif model == 'RADIAL':
        # params: f, cx, cy, k1, k2
        f, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0, 0], dtype=np.float64)

    elif model in ('OPENCV', 'FULL_OPENCV'):
        # params: fx, fy, cx, cy, k1, k2, p1, p2 [, k3 …]
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k1, k2, p1, p2 = p[4], p[5], p[6], p[7]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2], dtype=np.float64)

    else:
        # Generic fallback — treat as PINHOLE
        print_warning(f"Unknown camera model '{model}', treating as PINHOLE.")
        fx = p[0]
        fy = p[1] if len(p) >= 4 else p[0]
        cx = p[-2]
        cy = p[-1]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4)

    return K, dist, w, h


def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,    1 - 2*qx**2 - 2*qy**2],
    ], dtype=np.float64)


def image_pose(colmap_image):
    """
    Return the world-to-camera rotation matrix R (3x3) and translation t (3,)
    from a pycolmap Image object.

    COLMAP convention: P_cam = R @ P_world + t
    Camera centre in world: C = -R.T @ t

    Handles multiple pycolmap API versions:
      - cam_from_world as property  → Rigid3d with .rotation.matrix() / .translation
      - cam_from_world as method()  → same Rigid3d, but must be called first
      - qvec / tvec                 → older pycolmap (< 0.4)
    """
    # 1. cam_from_world property (most common in pycolmap >= 0.4)
    cfw_attr = getattr(colmap_image, 'cam_from_world', None)
    if cfw_attr is not None:
        # If it's a callable (method) rather than a property, call it
        cfw = cfw_attr() if callable(cfw_attr) else cfw_attr
        try:
            rot = cfw.rotation
            # rotation may itself be callable (returns Rotation3d) or already Rotation3d
            if callable(rot):
                rot = rot()
            R = np.array(rot.matrix(), dtype=np.float64)
            t = np.array(cfw.translation, dtype=np.float64)
            return R, t
        except AttributeError:
            pass   # fall through to qvec

    # 2. qvec / tvec (pycolmap < 0.4)
    qvec = getattr(colmap_image, 'qvec', None)
    tvec = getattr(colmap_image, 'tvec', None)
    if qvec is not None and tvec is not None:
        return qvec_to_rotmat(qvec), np.array(tvec, dtype=np.float64)

    raise AttributeError(
        f"Cannot extract pose from pycolmap Image. "
        f"Available attributes: {[a for a in dir(colmap_image) if not a.startswith('_')]}"
    )


def camera_centre(R, t):
    """Camera centre in world coordinates: C = -R.T @ t"""
    return (-R.T @ t)

##############################################################################
# STEP 1 — LOAD SFM MODEL
##############################################################################

def load_sfm():
    if not os.path.exists(SFM_RECONSTRUCTION_PATH):
        print_error(f"SfM reconstruction not found at {SFM_RECONSTRUCTION_PATH}. Run pipeline.py first.")
        return None
    recon = pycolmap.Reconstruction(SFM_RECONSTRUCTION_PATH)
    n_cams = len(recon.cameras)
    n_imgs = len(recon.images)
    n_pts  = len(recon.points3D)
    print_info(f"Loaded SfM: {n_cams} camera(s), {n_imgs} image(s), {n_pts} sparse 3D points.")
    return recon

##############################################################################
# STEP 2 — FIND NEAREST NEIGHBOURS
##############################################################################

def find_neighbours(recon, k):
    """
    For each registered image, find the k closest images by camera centre
    (Euclidean distance in world space).

    Returns dict: image_id → list of neighbour image_ids (closest first).
    """
    ids = list(recon.images.keys())
    centres = []
    for iid in ids:
        img = recon.images[iid]
        R, t = image_pose(img)
        centres.append(camera_centre(R, t))

    centres_arr = np.array(centres)   # (N, 3)
    tree = cKDTree(centres_arr)
    _, nn_indices = tree.query(centres_arr, k=k + 1)  # +1: first match is self

    neighbours = {}
    for i, iid in enumerate(ids):
        # skip index 0 (self), take up to k
        neighbours[iid] = [ids[nn_indices[i, j]] for j in range(1, k + 1) if nn_indices[i, j] < len(ids)]
    return neighbours

##############################################################################
# STEP 3 — STEREO DEPTH ESTIMATION
##############################################################################

def load_image_gray(name):
    path = os.path.join(IMAGES_PATH, name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img

def load_image_color(name):
    path = os.path.join(IMAGES_PATH, name)
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def compute_stereo_depth(img1_gray, img2_gray,
                          K1, dist1, w1, h1,
                          K2, dist2, w2, h2,
                          R1_world, t1, R2_world, t2):
    """
    Given two images (reference=1, neighbour=2) with their camera parameters,
    compute a depth map for image 1 using stereo SGBM.

    Returns depth map (h1 x w1) in metres, float32. Invalid pixels → 0.
    """
    # Relative pose: 2-to-1 in camera coordinates
    # P_cam1 = R1 @ P_world + t1
    # P_cam2 = R2 @ P_world + t2
    # => R_rel = R2 @ R1.T,  t_rel = t2 - R_rel @ t1
    R_rel = R2_world @ R1_world.T
    t_rel = t2 - R_rel @ t1

    # Undistort + stereo rectify
    R_rect1, R_rect2, P_rect1, P_rect2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2,
        (w1, h1), R_rel, t_rel.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R_rect1, P_rect1, (w1, h1), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R_rect2, P_rect2, (w2, h2), cv2.CV_32FC1)

    rect1 = cv2.remap(img1_gray, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2_gray, map2x, map2y, cv2.INTER_LINEAR)

    # StereoSGBM — semi-global block matching (CPU, no GPU needed)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK_SIZE,
        P1=SGBM_P1,
        P2=SGBM_P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp_fixed = sgbm.compute(rect1, rect2)   # int16, fixed-point ×16

    # Convert to float disparity
    disp = disp_fixed.astype(np.float32) / 16.0
    disp[disp <= 0] = 0

    # Baseline and focal length from Q matrix
    # Q = [[1,0,0,-cx], [0,1,0,-cy], [0,0,0,f], [0,0,-1/B,cx2-cx/B]]
    f  = Q[2, 3]
    Tx = -1.0 / Q[3, 2]   # baseline in world units (metres if t was in metres)

    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disp > 0, (f * abs(Tx)) / disp, 0.0)

    # Clip unreliable depths
    depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0.0

    # The depth is in the rectified frame; we need it back in the original
    # image 1 frame. We warp the depth map through the inverse rectification.
    # Simplification: back-project using undistorted intrinsics.
    # (For sub-pixel accurate results a full re-projection is needed, but
    # this gives good dense coverage.)

    return depth.astype(np.float32), map1x, map1y, R_rect1, P_rect1

##############################################################################
# STEP 4 — BACK-PROJECT DEPTH MAP TO COLOURED 3D POINTS
##############################################################################

def depth_to_pointcloud(depth, color_bgr, K, R_world, t):
    """
    Given a depth map and the colour image (both in the undistorted/rectified
    image frame), back-project valid pixels to 3D world coordinates.

    Returns (pts_world [N,3], colors_rgb [N,3] float32 in [0,1]).
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    valid = depth > 0
    d = depth[valid]
    x_cam = (uu[valid] - cx) * d / fx
    y_cam = (vv[valid] - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)   # (N, 3)

    # Camera-to-world: P_world = R.T @ (P_cam - t)
    pts_world = (R_world.T @ (pts_cam - t).T).T          # (N, 3)

    # Colours
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colors = color_rgb[valid]

    return pts_world.astype(np.float32), colors

##############################################################################
# STEP 5 — BUILD FULL DENSE POINT CLOUD
##############################################################################

def build_dense_cloud(recon, neighbours):
    all_pts = []
    all_colors = []

    images_sorted = sorted(recon.images.keys())
    n_total = len(images_sorted)

    for idx, ref_id in enumerate(images_sorted):
        ref_img = recon.images[ref_id]
        cam = recon.cameras[ref_img.camera_id]
        K1, dist1, w1, h1 = camera_to_opencv(cam)
        R1, t1 = image_pose(ref_img)

        try:
            gray1 = load_image_gray(ref_img.name)
            color1 = load_image_color(ref_img.name)
        except FileNotFoundError as e:
            print_warning(str(e))
            continue

        # Resize to match SfM camera dimensions if needed
        if gray1.shape[1] != w1 or gray1.shape[0] != h1:
            gray1  = cv2.resize(gray1,  (w1, h1))
            color1 = cv2.resize(color1, (w1, h1))

        depth_stack = []
        ref_K_undist = None

        for nbr_id in neighbours.get(ref_id, []):
            nbr_img = recon.images[nbr_id]
            nbr_cam = recon.cameras[nbr_img.camera_id]
            K2, dist2, w2, h2 = camera_to_opencv(nbr_cam)
            R2, t2 = image_pose(nbr_img)

            try:
                gray2 = load_image_gray(nbr_img.name)
            except FileNotFoundError:
                continue

            if gray2.shape[1] != w2 or gray2.shape[0] != h2:
                gray2 = cv2.resize(gray2, (w2, h2))

            try:
                depth, map1x, map1y, R_rect1, P_rect1 = compute_stereo_depth(
                    gray1, gray2,
                    K1, dist1, w1, h1,
                    K2, dist2, w2, h2,
                    R1, t1, R2, t2
                )
                depth_stack.append(depth)
                # Build undistorted intrinsics from P_rect1 for back-projection
                if ref_K_undist is None:
                    ref_K_undist = np.array([
                        [P_rect1[0, 0], 0,            P_rect1[0, 2]],
                        [0,             P_rect1[1, 1], P_rect1[1, 2]],
                        [0,             0,             1            ]
                    ], dtype=np.float64)
            except Exception as e:
                print_warning(f"Stereo pair ({ref_img.name}, {nbr_img.name}) failed: {e}")
                continue

        if not depth_stack:
            continue

        # Median-fuse depth estimates from all neighbours
        depth_fused = np.median(np.stack(depth_stack, axis=0), axis=0).astype(np.float32)

        # Undistort colour image to match the rectified frame
        if ref_K_undist is None:
            ref_K_undist = K1
        color_undist = cv2.undistort(color1, K1, dist1, newCameraMatrix=ref_K_undist.astype(np.float32))

        pts, cols = depth_to_pointcloud(depth_fused, color_undist, ref_K_undist, R1, t1)

        if len(pts) > 0:
            all_pts.append(pts)
            all_colors.append(cols)

        print_info(f"[{idx+1}/{n_total}] {ref_img.name}: {len(pts)} points")

    if not all_pts:
        print_error("No points generated. Check image paths and SfM model.")
        return None

    pts_merged   = np.concatenate(all_pts,   axis=0)
    colors_merged = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_merged.astype(np.float64))

    print_success(f"Total dense points before filtering: {len(pcd.points):,}")
    return pcd

##############################################################################
# STEP 6 — OUTLIER REMOVAL + VOXEL DOWNSAMPLE
##############################################################################

def filter_cloud(pcd):
    # Voxel downsample to reduce redundancy
    voxel_size = 0.005   # 5 mm; adjust to scene scale
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print_info(f"After voxel downsample ({voxel_size} m): {len(pcd_down.points):,} points")

    # Statistical outlier removal
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    pcd_clean = pcd_down.select_by_index(ind)
    print_info(f"After statistical outlier removal: {len(pcd_clean.points):,} points")

    return pcd_clean

##############################################################################
# STEP 7 — SURFACE RECONSTRUCTION
##############################################################################

def reconstruct_mesh(pcd):
    print_info("Estimating normals…")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    print_info(f"Running Poisson reconstruction (depth={POISSON_DEPTH})…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, n_threads=-1   # n_threads=-1 → all CPU cores
    )

    # Trim low-density regions (artefacts at the boundary)
    density_threshold = np.percentile(np.asarray(densities), 5)
    vertices_to_remove = np.asarray(densities) < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    print_success(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh

##############################################################################
# MAIN
##############################################################################

def run(dataset_path, dataset_reset=None):
    global DATASET_PATH, DATASET_RESET, IMAGES_PATH, SFM_PATH, SFM_RECONSTRUCTION_PATH, DENSE_PATH
    DATASET_PATH = dataset_path
    if dataset_reset is not None:
        DATASET_RESET = dataset_reset
    IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
    SFM_PATH = os.path.join(DATASET_PATH, 'sfm')
    SFM_RECONSTRUCTION_PATH = os.path.join(SFM_PATH, '0')
    DENSE_PATH = os.path.join(DATASET_PATH, 'dense')

    time_start = time.time()

    if DATASET_RESET and os.path.exists(DENSE_PATH):
        shutil.rmtree(DENSE_PATH)
        print_info(f"Reset: deleted {DENSE_PATH}")

    os.makedirs(DENSE_PATH, exist_ok=True)

    fused_ply_path = os.path.join(DENSE_PATH, 'fused.ply')
    mesh_ply_path  = os.path.join(DENSE_PATH, 'meshed-poisson.ply')

    print_step("Load SfM reconstruction")
    recon = load_sfm()
    if recon is None:
        return

    print_step(f"Find {K_NEIGHBOURS} nearest neighbours per view")
    neighbours = find_neighbours(recon, K_NEIGHBOURS)

    print_step("Multi-view stereo (StereoSGBM, CPU-only)")
    pcd_raw = build_dense_cloud(recon, neighbours)
    if pcd_raw is None:
        return

    print_step("Outlier removal & downsample")
    pcd_clean = filter_cloud(pcd_raw)

    o3d.io.write_point_cloud(fused_ply_path, pcd_clean)
    print_success(f"Dense point cloud saved: {fused_ply_path}")

    print_step("Poisson surface reconstruction")
    mesh = reconstruct_mesh(pcd_clean)
    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
    print_success(f"Mesh saved: {mesh_ply_path}")

    elapsed = time.time() - time_start
    print_step("Summary")
    print_info(f"Output directory : {DENSE_PATH}")
    print_info(f"Dense cloud      : {fused_ply_path}")
    print_info(f"Mesh             : {mesh_ply_path}")
    print_info(f"Total time       : {elapsed:.1f} s")


if __name__ == "__main__":
    run(DATASET_PATH)
