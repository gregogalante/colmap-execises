import os
import time
import subprocess
import shutil

##############################################################################
# CONFIG
##############################################################################

DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
DATASET_PATH = os.path.join(DATASETS_PATH, 'banana')  # Same dataset as pipeline.py
DATASET_RESET = False  # Set to True to redo dense reconstruction from scratch

IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
SFM_PATH = os.path.join(DATASET_PATH, 'sfm')
SFM_RECONSTRUCTION_PATH = os.path.join(SFM_PATH, '0')
DENSE_PATH = os.path.join(DATASET_PATH, 'dense')

# CPU-only: set to -1 to disable GPU in PatchMatch Stereo
PATCHMATCH_GPU_INDEX = -1

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
# HELPERS
##############################################################################

def run_colmap(args: list[str]) -> bool:
    """Run a COLMAP command. Returns True on success, False on failure."""
    cmd = ["colmap"] + args
    print_info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print_error(f"COLMAP command failed with exit code {result.returncode}.")
        return False
    return True

def check_colmap_available() -> bool:
    try:
        result = subprocess.run(["colmap", "help"], capture_output=True)
        return True
    except FileNotFoundError:
        return False

##############################################################################
# RESET
##############################################################################

def reset_dense():
    if DATASET_RESET and os.path.exists(DENSE_PATH):
        shutil.rmtree(DENSE_PATH)
        print_info(f"Deleted existing dense path: {DENSE_PATH}")

##############################################################################
# STEP 1 — IMAGE UNDISTORTION
# Undistorts images and exports the sparse model in a format suitable for MVS.
# Output: dense/images/, dense/sparse/, dense/stereo/
##############################################################################

def undistort_images():
    undistorted_images_path = os.path.join(DENSE_PATH, 'images')
    if os.path.exists(undistorted_images_path) and os.listdir(undistorted_images_path):
        print_info("Undistorted images already exist. Skipping.")
        return True

    if not os.path.exists(SFM_RECONSTRUCTION_PATH):
        print_error(f"SFM reconstruction not found at {SFM_RECONSTRUCTION_PATH}. Run pipeline.py first.")
        return False

    os.makedirs(DENSE_PATH, exist_ok=True)

    ok = run_colmap([
        "image_undistorter",
        "--image_path", IMAGES_PATH,
        "--input_path", SFM_RECONSTRUCTION_PATH,
        "--output_path", DENSE_PATH,
        "--output_type", "COLMAP",
        "--max_image_size", "1000",
    ])
    if not ok:
        return False

    print_success("Image undistortion completed.")
    return True

##############################################################################
# STEP 2 — PATCH MATCH STEREO (CPU)
# Computes per-image depth maps and normal maps using PatchMatch.
# --PatchMatchStereo.gpu_index -1 forces CPU execution (slow but GPU-free).
# Output: dense/stereo/depth_maps/, dense/stereo/normal_maps/
##############################################################################

def patch_match_stereo():
    depth_maps_path = os.path.join(DENSE_PATH, 'stereo', 'depth_maps')
    if os.path.exists(depth_maps_path) and os.listdir(depth_maps_path):
        print_info("Depth maps already exist. Skipping PatchMatch Stereo.")
        return True

    print_warning("PatchMatch Stereo in CPU mode is slow. This may take a long time.")

    ok = run_colmap([
        "patch_match_stereo",
        "--workspace_path", DENSE_PATH,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", str(PATCHMATCH_GPU_INDEX),
        "--PatchMatchStereo.depth_min", "0.01",
        "--PatchMatchStereo.depth_max", "100.0",
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.num_samples", "15",
        "--PatchMatchStereo.num_iterations", "5",
        "--PatchMatchStereo.geom_consistency", "true",
    ])
    if not ok:
        return False

    print_success("PatchMatch Stereo completed.")
    return True

##############################################################################
# STEP 3 — STEREO FUSION
# Fuses all depth maps into a single dense colored point cloud.
# Output: dense/fused.ply
##############################################################################

def stereo_fusion():
    fused_ply_path = os.path.join(DENSE_PATH, 'fused.ply')
    if os.path.exists(fused_ply_path):
        print_info("Fused point cloud already exists. Skipping Stereo Fusion.")
        return True

    ok = run_colmap([
        "stereo_fusion",
        "--workspace_path", DENSE_PATH,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply_path,
        "--StereoFusion.min_num_pixels", "3",
        "--StereoFusion.max_reproj_error", "2.0",
        "--StereoFusion.max_depth_error", "0.01",
    ])
    if not ok:
        return False

    print_success(f"Stereo fusion completed. Dense point cloud: {fused_ply_path}")
    return True

##############################################################################
# STEP 4 — POISSON SURFACE MESHING
# Reconstructs a watertight triangle mesh from the fused point cloud using
# Screened Poisson reconstruction.
# Output: dense/meshed-poisson.ply
##############################################################################

def poisson_mesh():
    fused_ply_path = os.path.join(DENSE_PATH, 'fused.ply')
    meshed_ply_path = os.path.join(DENSE_PATH, 'meshed-poisson.ply')

    if os.path.exists(meshed_ply_path):
        print_info("Poisson mesh already exists. Skipping.")
        return True

    if not os.path.exists(fused_ply_path):
        print_error(f"Fused point cloud not found at {fused_ply_path}. Run stereo_fusion first.")
        return False

    ok = run_colmap([
        "poisson_mesher",
        "--input_path", fused_ply_path,
        "--output_path", meshed_ply_path,
        "--PoissonMeshing.trim", "10",
    ])
    if not ok:
        print_warning("Poisson mesher failed. Trying Delaunay mesher as fallback...")
        return delaunay_mesh()

    print_success(f"Poisson mesh completed: {meshed_ply_path}")
    return True

##############################################################################
# STEP 4b — DELAUNAY SURFACE MESHING (fallback)
# Alternative surface reconstruction using Delaunay triangulation.
# Output: dense/meshed-delaunay.ply
##############################################################################

def delaunay_mesh():
    fused_ply_path = os.path.join(DENSE_PATH, 'fused.ply')
    meshed_ply_path = os.path.join(DENSE_PATH, 'meshed-delaunay.ply')

    if os.path.exists(meshed_ply_path):
        print_info("Delaunay mesh already exists. Skipping.")
        return True

    ok = run_colmap([
        "delaunay_mesher",
        "--input_path", DENSE_PATH,
        "--input_type", "dense",
        "--output_path", meshed_ply_path,
    ])
    if not ok:
        return False

    print_success(f"Delaunay mesh completed: {meshed_ply_path}")
    return True

##############################################################################
# STEP 5 — PRINT SUMMARY
##############################################################################

def print_summary():
    print_info("Dense reconstruction outputs:")
    outputs = [
        ("Undistorted images",   os.path.join(DENSE_PATH, 'images')),
        ("Sparse (undistorted)", os.path.join(DENSE_PATH, 'sparse')),
        ("Depth maps",           os.path.join(DENSE_PATH, 'stereo', 'depth_maps')),
        ("Fused point cloud",    os.path.join(DENSE_PATH, 'fused.ply')),
        ("Poisson mesh",         os.path.join(DENSE_PATH, 'meshed-poisson.ply')),
        ("Delaunay mesh",        os.path.join(DENSE_PATH, 'meshed-delaunay.ply')),
    ]
    for label, path in outputs:
        exists = os.path.exists(path)
        status = "\033[92m[OK]\033[0m" if exists else "\033[90m[--]\033[0m"
        print(f"  {status}  {label}: {path}")

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
    time_start = time.time()

    if not check_colmap_available():
        print_error("'colmap' binary not found in PATH. Install COLMAP: https://colmap.github.io/install.html")
        exit(1)

    reset_dense()

    print_step("Undistort Images")
    if not undistort_images():
        print_error("Undistortion failed. Aborting.")
        exit(1)

    print_step("PatchMatch Stereo (CPU)")
    if not patch_match_stereo():
        print_error("PatchMatch Stereo failed. Aborting.")
        exit(1)

    print_step("Stereo Fusion")
    if not stereo_fusion():
        print_error("Stereo Fusion failed. Aborting.")
        exit(1)

    print_step("Poisson Surface Meshing")
    poisson_mesh()  # non-fatal: fallback to Delaunay internally

    time_end = time.time()
    print_step("Summary")
    print_summary()
    print_info(f"Total execution time: {time_end - time_start:.2f} seconds")
