import os
import multiprocessing as mp
import numpy as np
import h5py
from tqdm.auto import tqdm
from numba import njit
from read_write_model import Camera, Image, read_images_binary, read_cameras_binary

UNDISTORTED_ROOT = '/cvlabdata1/cvlab/datasets_tyszkiew/megadepth/undistorted'

def clean_image(image: Image) -> Image:
    mask = image.point3D_ids != -1
    ids = image.point3D_ids[mask]
    xys = image.xys[mask]

    order = np.argsort(ids)
    ids = np.ascontiguousarray(ids[order])
    xys = np.ascontiguousarray(xys[order])

    return image._replace(point3D_ids=ids, xys=xys)

@njit
def point_overlap(a: np.ndarray, b: np.ndarray) -> int:
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

def camera_to_K(camera: Camera) -> np.ndarray:
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

def reindex_images(images: dict[int, Image]) -> list[Image]:
    '''
    Reindex images so that the keys are contiguous
    '''
    image_ids = list(images.keys())
    image_ids.sort()
    image_ids = np.array(image_ids)

    new_image_ids = np.arange(len(image_ids))
    new_to_old_id = dict(zip(new_image_ids, image_ids))

    new_images = []
    for new_image_id in new_image_ids:
        image = images[new_to_old_id[new_image_id]]
        image = image._replace(id=new_image_id)
        new_images.append(image)
    
    return new_images

def process_model(scene: str, model: str):
    path = f'{UNDISTORTED_ROOT}/{scene}/{model}'

    images = read_images_binary(f'{path}/sparse/images.bin')
    cameras = read_cameras_binary(f'{path}/sparse/cameras.bin')

    images = {k: clean_image(v) for k, v in images.items()}
    images = reindex_images(images)

    # create image metadata
    image_metadata = []
    for image in images:
        metadata = dict(
            R=image.qvec2rotmat(),
            t=image.tvec,
            K=camera_to_K(cameras[image.camera_id]),
            image_name=image.name.encode('utf-8'),
        )
        image_metadata.append(metadata)
    # convert to dict of numpy arrays
    image_metadata = {k: np.array([v[k] for v in image_metadata]) for k in image_metadata[0].keys()}

    # filter out images with weird aspect ratios
    images_filtered = []
    for image in images:
        camera = cameras[image.camera_id]
        h, w = camera.height, camera.width
        if h / w > 1.55 or w / h > 1.55:
            continue
        images_filtered.append(image)

    # create pair list
    pairs, overlaps = [], []
    for i, image_i in enumerate(images_filtered):
        for j, image_j in enumerate(images_filtered):
            if i < j:
                overlap = point_overlap(image_i.point3D_ids, image_j.point3D_ids)
                if overlap > 0:
                    pairs.append((image_i.id, image_j.id))
                    overlaps.append(overlap)

    pairs = np.array(pairs, dtype=np.uint16)
    overlaps = np.array(overlaps, dtype=np.uint16)

    with h5py.File(f'{path}/metadata.h5', 'w') as f:
        f.create_dataset('pairs', data=pairs)
        f.create_dataset('overlaps', data=overlaps)
        for k, v in image_metadata.items():
            f.create_dataset(k, data=v)

def _job(args):
    try:
        process_model(*args)
    except Exception as e:
        print(f'Failed to process {args} due to {e}')

if __name__ == '__main__':
    args = []
    scenes = os.listdir(UNDISTORTED_ROOT)
    for scene in scenes:
        models = os.listdir(f'{UNDISTORTED_ROOT}/{scene}')
        for model in models:
            args.append((scene, model))
    
    with mp.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(_job, args), total=len(args)):
            pass