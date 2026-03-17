import os
import time
import shutil
import pycolmap
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

from libs.read_write_model import read_cameras_binary, write_cameras_text, read_images_binary, write_images_text

IMAGE_MAX_DIMENSION = 1024

DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'datasets')

# Change configs
# -----------------------------------------------------------------------------

DATASET_PATH = os.path.join(DATASETS_PATH, 'over-office-1')  # Change this to your dataset name (e.g., 'building1', 'building2', etc.)
DATASET_RESET = True  # Set to True to reset the dataset by deleting existing images, database, and SFM reconstruction

# -----------------------------------------------------------------------------

TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
DATABASE_PATH = os.path.join(DATASET_PATH, 'database.db')
SFM_PATH = os.path.join(DATASET_PATH, 'sfm')

if DATASET_RESET:
  if os.path.exists(IMAGES_PATH):
    shutil.rmtree(IMAGES_PATH)
  if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
  if os.path.exists(SFM_PATH):
    shutil.rmtree(SFM_PATH)

##############################################################################
# PRINT HELPERS
##############################################################################

def print_error(message):
  color = "\033[91m"  # Red color
  reset = "\033[0m"
  print(f"{color}ERROR: {message}{reset}")

def print_success(message):
  color = "\033[92m"  # Green color
  reset = "\033[0m"
  print(f"{color}SUCCESS: {message}{reset}")

def print_info(message):
  color = "\033[96m"  # Cyan color
  reset = "\033[0m"
  print(f"{color}INFO: {message}{reset}")

def print_warning(message):
  color = "\033[93m"  # Yellow color
  reset = "\033[0m"
  print(f"{color}WARNING: {message}{reset}")

def print_step(message):
  color = "\033[94m"  # Blue color
  reset = "\033[0m"
  print("\n")
  print(f"{color}{'='*100}{reset}")
  print(f"{color}RUNNING STEP: {message}{reset}")
  print(f"{color}{'='*100}{reset}\n")

##############################################################################
# BUILD IMAGES
##############################################################################

def build_images():
  if os.path.exists(IMAGES_PATH) and os.listdir(IMAGES_PATH):
    print_info(f"Images path {IMAGES_PATH} already exists and is not empty. Skipping image build.")
    return
  
  if not os.path.exists(TRAIN_PATH):
    print_error(f"Train path {TRAIN_PATH} does not exist. Cannot build images.")
    return
  
  os.makedirs(IMAGES_PATH, exist_ok=True)

  def process_image(filename):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
      print_warning(f"Skipping non-image file: {filename}")
      return
    source_path = os.path.join(TRAIN_PATH, filename)
    dest_path = os.path.join(IMAGES_PATH, filename)
    try:
      with Image.open(source_path) as img:
        width, height = img.size
        max_dimension = max(width, height)
        if max_dimension > IMAGE_MAX_DIMENSION:
          scale = IMAGE_MAX_DIMENSION / max_dimension
          new_size = (int(width * scale), int(height * scale))
          img = img.resize(new_size, Image.Resampling.LANCZOS)
          print_info(f"Resized image {filename} from ({width}, {height}) to {new_size}.")
        img.save(dest_path)
        print_success(f"Copied image {filename} to images path.")
    except Exception as e:
      print_error(f"Failed to process image {filename}: {e}")

  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_image, f): f for f in os.listdir(TRAIN_PATH)}
    for future in as_completed(futures):
      future.result()

##############################################################################
# EXTRACT FEATURES
##############################################################################

def extract_features():
  if not os.path.exists(IMAGES_PATH) or not os.listdir(IMAGES_PATH):
    print_error(f"Images path {IMAGES_PATH} does not exist or is empty. Cannot extract features.")
    return

  print_info("Extracting features using pycolmap...")
  pycolmap.extract_features(DATABASE_PATH, IMAGES_PATH)
  print_success("Feature extraction completed.")

##############################################################################
# MATCH FEATURES
##############################################################################

def match_features():
  if not os.path.exists(DATABASE_PATH):
    print_error(f"Database path {DATABASE_PATH} does not exist. Cannot perform matching.")
    return

  print_info("Performing exhaustive matching using pycolmap...")
  pycolmap.match_exhaustive(DATABASE_PATH)
  print_success("Exhaustive matching completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION
##############################################################################

def build_sfm_reconstruction():
  if not os.path.exists(IMAGES_PATH) or not os.listdir(IMAGES_PATH):
    print_error(f"Images path {IMAGES_PATH} does not exist or is empty. Cannot build SFM reconstruction.")
    return

  sfm_reconstruction_path = os.path.join(SFM_PATH, "0")
  if os.path.exists(sfm_reconstruction_path) and os.listdir(sfm_reconstruction_path):
    print_info(f"SFM reconstruction path {sfm_reconstruction_path} already exists and is not empty. Skipping SFM reconstruction.")
    return
  
  if not os.path.exists(SFM_PATH):
    os.makedirs(SFM_PATH, exist_ok=True)

  print_info("Running incremental mapping to build SFM reconstruction using pycolmap...")
  reconstructions = pycolmap.incremental_mapping(DATABASE_PATH, IMAGES_PATH, SFM_PATH)
  reconstruction = reconstructions[0]
  print(reconstruction.summary())
  print_success("SFM reconstruction completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION PLY
##############################################################################

def build_sfm_reconstruction_ply():
  if not os.path.exists(SFM_PATH):
    print_error(f"SFM path {SFM_PATH} does not exist. Cannot build SFM reconstruction PLY.")
    return

  sfm_reconstruction_path = os.path.join(SFM_PATH, "0")
  sfm_reconstruction_ply_path = os.path.join(SFM_PATH, "reconstruction.ply")
  if os.path.exists(sfm_reconstruction_ply_path):
    print_info(f"SFM reconstruction PLY path {sfm_reconstruction_ply_path} already exists. Skipping PLY export.")
    return
  print_info("Exporting SFM reconstruction to PLY using pycolmap...")
  reconstruction = pycolmap.Reconstruction(sfm_reconstruction_path)
  reconstruction.export_PLY(sfm_reconstruction_ply_path)
  print_success(f"SFM reconstruction exported to {sfm_reconstruction_ply_path}.")

##############################################################################
# BUILD SFM RECONSTRUCTION TXT
##############################################################################

def build_sfm_reconstruction_txt():
  if not os.path.exists(SFM_PATH):
    print_error(f"SFM path {SFM_PATH} does not exist. Cannot build SFM reconstruction TXT.")
    return

  sfm_reconstruction_path = os.path.join(SFM_PATH, "0")
  cameras_bin_path = os.path.join(sfm_reconstruction_path, "cameras.bin")
  cameras_txt_path = os.path.join(sfm_reconstruction_path, "cameras.txt")
  images_bin_path = os.path.join(sfm_reconstruction_path, "images.bin")
  images_txt_path = os.path.join(sfm_reconstruction_path, "images.txt")

  if os.path.exists(cameras_txt_path) and os.path.exists(images_txt_path):
    print_info(f"SFM reconstruction TXT files already exist. Skipping TXT export.")
    return

  print_info("Converting SFM reconstruction from BIN to TXT...")
  cameras = read_cameras_binary(cameras_bin_path) 
  write_cameras_text(cameras, cameras_txt_path)
  images = read_images_binary(images_bin_path)
  write_images_text(images, images_txt_path)
  print_success(f"SFM reconstruction cameras exported to {cameras_txt_path}.")
  print_success(f"SFM reconstruction images exported to {images_txt_path}.")

##############################################################################
# BUILD SFM RECONSTRUCTION transforms.json
##############################################################################

def build_sfm_reconstruction_transforms_json():
  if not os.path.exists(SFM_PATH):
    print_error(f"SFM path {SFM_PATH} does not exist. Cannot build SFM reconstruction transforms.json.")
    return

  sfm_reconstruction_path = os.path.join(SFM_PATH, "0")
  sfm_transforms_json_path = os.path.join(SFM_PATH, "transforms.json")

  if os.path.exists(sfm_transforms_json_path):
    print_info(f"SFM reconstruction transforms.json path {sfm_transforms_json_path} already exists. Skipping transforms.json export.")
    return

  print_info("Exporting SFM reconstruction to transforms.json using colmap2nerf...")
  command = f"python colmap2nerf.py \
    --colmap_matcher exhaustive \
    --aabb_scale 16 \
    --images {IMAGES_PATH} \
    --text {sfm_reconstruction_path} \
    --out {sfm_transforms_json_path} \
  "
  os.system(command)
  
  if not os.path.exists(sfm_transforms_json_path):
    print_error(f"Failed to export transforms.json to {sfm_transforms_json_path}.")
    return
  print_success(f"SFM reconstruction exported to {sfm_transforms_json_path}.")

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
  time_start = time.time()
  print_step("🚀 Build Images")
  build_images()
  print_step("🚀 Extract Features")
  extract_features()
  print_step("🚀 Match Features")
  match_features()
  print_step("🚀 Build SFM Reconstruction")
  build_sfm_reconstruction()
  print_step("🚀 Build SFM Reconstruction PLY")
  build_sfm_reconstruction_ply()
  print_step("🚀 Build SFM Reconstruction TXT")
  build_sfm_reconstruction_txt()
  print_step("🚀 Build SFM Reconstruction transforms.json")
  build_sfm_reconstruction_transforms_json()

  print_step("✅ Pipeline completed")
  time_end = time.time()
  time_total = time_end - time_start
  print_success(f"Total execution time: {time_total:.2f} seconds")
