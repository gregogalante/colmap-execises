import os
import sys
import json
import time
import shutil
import argparse
import pycolmap
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

from libs.read_write_model import read_cameras_binary, write_cameras_text, read_images_binary, write_images_text, read_points3D_binary, write_points3D_text

IMAGE_MAX_DIMENSION = 1024

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
# ARGS
##############################################################################

def parse_args():
  parser = argparse.ArgumentParser(description="COLMAP SfM pipeline")
  parser.add_argument("--dataset", required=True, help="Path to dataset directory (e.g. datasets/home)")
  parser.add_argument("--reset", action="store_true", help="Reset the dataset by deleting existing images, database, and SFM reconstruction")
  return parser.parse_args()

##############################################################################
# BUILD IMAGES
##############################################################################

def build_images(train_path, images_path):
  if os.path.exists(images_path) and os.listdir(images_path):
    print_info(f"Images path {images_path} already exists and is not empty. Skipping image build.")
    return

  if not os.path.exists(train_path):
    print_error(f"Train path {train_path} does not exist. Cannot build images.")
    return

  os.makedirs(images_path, exist_ok=True)

  def process_image(filename):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
      print_warning(f"Skipping non-image file: {filename}")
      return
    source_path = os.path.join(train_path, filename)
    dest_path = os.path.join(images_path, filename)
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
    futures = {executor.submit(process_image, f): f for f in os.listdir(train_path)}
    for future in as_completed(futures):
      future.result()

##############################################################################
# EXTRACT FEATURES
##############################################################################

def extract_features(database_path, images_path):
  if not os.path.exists(images_path) or not os.listdir(images_path):
    print_error(f"Images path {images_path} does not exist or is empty. Cannot extract features.")
    return

  print_info("Extracting features using pycolmap...")
  pycolmap.extract_features(database_path, images_path)
  print_success("Feature extraction completed.")

##############################################################################
# MATCH FEATURES
##############################################################################

def match_features(database_path):
  if not os.path.exists(database_path):
    print_error(f"Database path {database_path} does not exist. Cannot perform matching.")
    return

  print_info("Performing exhaustive matching using pycolmap...")
  pycolmap.match_exhaustive(database_path)
  print_success("Exhaustive matching completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION
##############################################################################

def build_sfm_reconstruction(database_path, images_path, sfm_path):
  if not os.path.exists(images_path) or not os.listdir(images_path):
    print_error(f"Images path {images_path} does not exist or is empty. Cannot build SFM reconstruction.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  if os.path.exists(sfm_reconstruction_path) and os.listdir(sfm_reconstruction_path):
    print_info(f"SFM reconstruction path {sfm_reconstruction_path} already exists and is not empty. Skipping SFM reconstruction.")
    return

  if not os.path.exists(sfm_path):
    os.makedirs(sfm_path, exist_ok=True)

  print_info("Running incremental mapping to build SFM reconstruction using pycolmap...")
  reconstructions = pycolmap.incremental_mapping(database_path, images_path, sfm_path)
  reconstruction = reconstructions[0]
  print(reconstruction.summary())
  print_success("SFM reconstruction completed.")

##############################################################################
# BUILD SFM RECONSTRUCTION PLY
##############################################################################

def build_sfm_reconstruction_ply(sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction PLY.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  sfm_reconstruction_ply_path = os.path.join(sfm_path, "reconstruction.ply")
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

def build_sfm_reconstruction_txt(sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction TXT.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  cameras_bin_path = os.path.join(sfm_reconstruction_path, "cameras.bin")
  cameras_txt_path = os.path.join(sfm_reconstruction_path, "cameras.txt")
  images_bin_path = os.path.join(sfm_reconstruction_path, "images.bin")
  images_txt_path = os.path.join(sfm_reconstruction_path, "images.txt")
  points3D_bin_path = os.path.join(sfm_reconstruction_path, "points3D.bin")
  points3D_txt_path = os.path.join(sfm_reconstruction_path, "points3D.txt")

  if os.path.exists(cameras_txt_path) and os.path.exists(images_txt_path):
    print_info(f"SFM reconstruction TXT files already exist. Skipping TXT export.")
    return

  print_info("Converting SFM reconstruction from BIN to TXT...")
  cameras = read_cameras_binary(cameras_bin_path)
  write_cameras_text(cameras, cameras_txt_path)
  print_success(f"SFM reconstruction cameras exported to {cameras_txt_path}.")
  images = read_images_binary(images_bin_path)
  write_images_text(images, images_txt_path)
  print_success(f"SFM reconstruction images exported to {images_txt_path}.")
  points3D = read_points3D_binary(points3D_bin_path)
  write_points3D_text(points3D, points3D_txt_path)
  print_success(f"SFM reconstruction points3D exported to {points3D_txt_path}.")

##############################################################################
# BUILD SFM RECONSTRUCTION transforms.json
##############################################################################

def build_sfm_reconstruction_transforms_json(images_path, sfm_path):
  if not os.path.exists(sfm_path):
    print_error(f"SFM path {sfm_path} does not exist. Cannot build SFM reconstruction transforms.json.")
    return

  sfm_reconstruction_path = os.path.join(sfm_path, "0")
  sfm_transforms_json_path = os.path.join(sfm_path, "transforms.json")

  if os.path.exists(sfm_transforms_json_path):
    print_info(f"SFM reconstruction transforms.json path {sfm_transforms_json_path} already exists. Skipping transforms.json export.")
    return

  print_info("Exporting SFM reconstruction to transforms.json using colmap2nerf...")
  command = f"python colmap2nerf.py \
    --colmap_matcher exhaustive \
    --aabb_scale 16 \
    --images {images_path} \
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

def main():
  args = parse_args()

  dataset_path = args.dataset
  train_path = os.path.join(dataset_path, 'train')
  images_path = os.path.join(dataset_path, 'images')
  database_path = os.path.join(dataset_path, 'database.db')
  sfm_path = os.path.join(dataset_path, 'sfm')

  if not os.path.exists(dataset_path):
    print_error(f"Dataset path {dataset_path} does not exist.")
    sys.exit(1)

  config_path = os.path.join(dataset_path, 'config.json')
  config = {"image_max_dimension": IMAGE_MAX_DIMENSION}
  with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
  print_info(f"Config saved to {config_path}")

  if args.reset:
    if os.path.exists(images_path):
      shutil.rmtree(images_path)
    if os.path.exists(database_path):
      os.remove(database_path)
    if os.path.exists(sfm_path):
      shutil.rmtree(sfm_path)

  time_start = time.time()
  print_step("🚀 Build Images")
  build_images(train_path, images_path)
  print_step("🚀 Extract Features")
  extract_features(database_path, images_path)
  print_step("🚀 Match Features")
  match_features(database_path)
  print_step("🚀 Build SFM Reconstruction")
  build_sfm_reconstruction(database_path, images_path, sfm_path)
  print_step("🚀 Build SFM Reconstruction PLY")
  build_sfm_reconstruction_ply(sfm_path)
  print_step("🚀 Build SFM Reconstruction TXT")
  build_sfm_reconstruction_txt(sfm_path)
  print_step("🚀 Build SFM Reconstruction transforms.json")
  build_sfm_reconstruction_transforms_json(images_path, sfm_path)

  print_step("✅ Pipeline completed")
  time_end = time.time()
  time_total = time_end - time_start
  print_success(f"Total execution time: {time_total:.2f} seconds")

if __name__ == "__main__":
  main()
