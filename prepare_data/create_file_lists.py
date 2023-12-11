from __future__ import annotations
import os
import multiprocessing as mp
from dataclasses import dataclass
from itertools import combinations

import json
import numpy as np
import h5py
from numba import njit
from tqdm.auto import tqdm

@njit
def _point_overlap(a: np.ndarray, b: np.ndarray) -> int:
    n, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            n += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return n

@njit
def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

@dataclass(eq=True, frozen=True)
class CameraModel:
    model_id: int
    model_name: str
    num_params: int

@dataclass
class Camera:
    id: int
    model: CameraModel
    width: int
    height: int
    params: np.ndarray

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    n_features: int
    point3D_ids: np.ndarray # sorted

    def point_overlap(self, other: Image) -> int:
        return _point_overlap(self.point3D_ids, other.point3D_ids)

    def rotmat(self):
        return qvec2rotmat(self.qvec)

def read_images_text(path: str) -> dict[int, Image]:
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                n_features = len(point3D_ids)
                point3D_ids = point3D_ids[point3D_ids != -1]
                point3D_ids = point3D_ids[np.argsort(point3D_ids)]

                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    n_features=n_features,
                    point3D_ids=point3D_ids,
                )
    return images

def read_cameras_text(path: str) -> dict[int, Camera]:
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras

def camera_to_K(camera):
    '''
    Assembles the camera params (given as an unstructured list by COLMAP) into
    an intrinsics matrix
    '''
    assert camera.model == 'PINHOLE', camera.model

    fx, fy, cx, cy = camera.params

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float32)

@dataclass
class Calibration:
    R: np.ndarray
    T: np.ndarray
    K: np.ndarray

def create_calibration(image: Image, cameras: dict[int, Camera]) -> Calibration:
    camera = cameras[image.camera_id]
    return Calibration(
        R=image.rotmat(),
        T=image.tvec,
        K=camera_to_K(camera),
    )

def process_scene(subset_path: str, compacted_imgs_path: str):
    sfm_images = read_images_text(os.path.join(subset_path, "images.txt"))
    sfm_image_names = frozenset([image.name for image in sfm_images.values()])
    #with open(os.path.join(out_path, 'sfm_images.json'), 'w') as f:
    #    json.dump(sorted(list(sfm_image_names)), f)
    sfm_cameras = read_cameras_text(os.path.join(subset_path, 'cameras.txt'))

    compacted_img_names = frozenset(os.listdir(compacted_imgs_path))
    available_img_names = sorted(list(sfm_image_names.intersection(compacted_img_names)))

    missing_img_names = sorted(list(sfm_image_names.difference(compacted_img_names)))
    #with open(os.path.join(out_path, 'missing_images.json'), 'w') as f:
    #    json.dump(missing_img_names, f)
    
    if len(available_img_names) == 0:
        return
    
    image_name_to_available_id = {name: i for i, name in enumerate(available_img_names)}

    available_images = [image for image in sfm_images.values() if image.name in available_img_names]
    calibrations = [create_calibration(image, sfm_cameras) for image in available_images]

    pairs = []
    for (img1, img2) in combinations(available_images, 2):
        overlap = img1.point_overlap(img2)
        if overlap == 0:
            continue
        id1 = image_name_to_available_id[img1.name]
        id2 = image_name_to_available_id[img2.name]
        pairs.append((id1, id2))
    
    with h5py.File(os.path.join(subset_path, 'split_metadata_current.h5'), 'w') as f:
        f.create_dataset('pairs', data=np.array(pairs, dtype=np.uint16))
        f.create_dataset('R', data=np.array([calibration.R for calibration in calibrations], dtype=np.float32))
        f.create_dataset('T', data=np.array([calibration.T for calibration in calibrations], dtype=np.float32))
        f.create_dataset('K', data=np.array([calibration.K for calibration in calibrations], dtype=np.float32))
        f.create_dataset('images', data=np.array([name.encode('utf-8') for name in available_img_names]))
    print(f'Wrote {len(pairs)} pairs to {subset_path}.')


def _job(args):
    subset_path, compacted_imgs_path = args
    process_scene(subset_path, compacted_imgs_path)

def main():
    megadepth_root = '/cvlabdata1/cvlab/datasets_tyszkiew/megadepth/MegaDepth_v1_SfM/'
    compacted_root = '/cvlabdata1/cvlab/datasets_tyszkiew/compacted-datasets/megadepth/scenes'

    tasks = []
    for scene_id in os.listdir(megadepth_root):
        sparse_path = os.path.join(megadepth_root, scene_id, 'sparse/manhattan')
        for subset_id in os.listdir(sparse_path):
            subset_path = os.path.join(sparse_path, subset_id)

            compacted_imgs_path = os.path.join(compacted_root, scene_id, 'images')
            if not os.path.isdir(compacted_imgs_path):
                continue
            
            tasks.append((subset_path, compacted_imgs_path))
    
    #with mp.Pool() as pool:
    #    for _ in tqdm(pool.imap_unordered(_job, tasks), total=len(tasks)):
    #        pass
    _job(tasks[1])

if __name__ == '__main__':
    main()