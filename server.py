import math
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pycolmap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE = Path(__file__).parent / "storage"
DATASETS = STORAGE / "datasets"
RELOCATIONS = STORAGE / "relocations"


@lru_cache(maxsize=16)
def load_reconstruction(name: str):
    sfm_path = DATASETS / name / "sfm" / "0"
    if not sfm_path.exists():
        raise HTTPException(404, f"SfM data not found for {name}")
    return pycolmap.Reconstruction(str(sfm_path))


def c2w_from_image(img):
    cfw = img.cam_from_world()
    mat34 = cfw.matrix()  # 3x4
    mat44 = np.vstack([mat34, [0, 0, 0, 1]])
    return np.linalg.inv(mat44).tolist()


def c2w_from_relocation(rotation_matrix, camera_center):
    R = np.array(rotation_matrix)  # world-to-cam
    C = np.array(camera_center)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = C
    return c2w.tolist()


@app.get("/api/datasets")
def list_datasets():
    names = sorted(d.name for d in DATASETS.iterdir() if d.is_dir())
    return {"datasets": names}


@app.get("/api/datasets/{name}/cameras")
def get_cameras(name: str):
    recon = load_reconstruction(name)
    cameras = []
    for img in recon.images.values():
        if not img.has_pose:
            continue
        cam = recon.cameras[img.camera_id]
        fl = cam.focal_length
        fovx = 2 * math.atan(cam.width / (2 * fl)) * 180 / math.pi
        fovy = 2 * math.atan(cam.height / (2 * fl)) * 180 / math.pi
        cameras.append({
            "image_name": img.name,
            "camera_to_world": c2w_from_image(img),
            "width": cam.width,
            "height": cam.height,
            "fovx": fovx,
            "fovy": fovy,
        })
    cameras.sort(key=lambda c: c["image_name"])
    return {"cameras": cameras}


@app.get("/api/datasets/{name}/images/{filename}")
def get_image(name: str, filename: str):
    path = DATASETS / name / "images" / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/datasets/{name}/reconstruction.ply")
def get_ply(name: str):
    path = DATASETS / name / "sfm" / "reconstruction.ply"
    if not path.exists():
        raise HTTPException(404, "PLY not found")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/api/relocations")
def list_relocations():
    relocations = []
    if not RELOCATIONS.exists():
        return {"relocations": []}
    for d in sorted(RELOCATIONS.iterdir()):
        if not d.is_dir():
            continue
        for jf in sorted(d.glob("*.json")):
            data = json.loads(jf.read_text())
            dataset_path = data.get("dataset", "")
            dataset_name = Path(dataset_path).name if dataset_path else ""
            entry = {
                "name": jf.stem,
                "folder": d.name,
                "dataset_name": dataset_name,
                "success": data.get("success", False),
                "num_inliers": data.get("num_inliers", 0),
                "num_correspondences": data.get("num_correspondences", 0),
                "camera_center": data.get("camera_center", [0, 0, 0]),
            }
            if data.get("rotation_matrix") and data.get("camera_center"):
                entry["camera_to_world"] = c2w_from_relocation(
                    data["rotation_matrix"], data["camera_center"]
                )
            relocations.append(entry)
    return {"relocations": relocations}


@app.get("/api/relocations/{folder}/{name}/image")
def get_relocation_image(folder: str, name: str):
    path = RELOCATIONS / folder / f"{name}.jpg"
    if not path.exists():
        raise HTTPException(404, "Relocation image not found")
    return FileResponse(path, media_type="image/jpeg")


UI_DIST = Path(__file__).parent / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
