import tarfile
import os
import re
import subprocess
import shutil
import time
from tqdm.auto import tqdm

COLMAP_EXE = '/cvlabdata2/home/tyszkiew/installs/colmap/git/colmap/build/src/exe/colmap'
path_re = re.compile(r'\./MegaDepth_v1_SfM/(\d+)')
unzip_path = '/cvlabdata1/cvlab/datasets_tyszkiew/megadepth/distorted'
undistort_path = '/cvlabdata1/cvlab/datasets_tyszkiew/megadepth/undistorted'

class Filter:
    def __init__(self):
        self.current_scene = None
        self.undistorted_scenes = frozenset(os.listdir(undistort_path))
        self.start_time = time.time()
        self.call_number = 0

    def __call__(self, member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
        self.call_number += 1

        if self.call_number % 1000 == 0:
            elapsed = time.time() - self.start_time
            print(f'Processed {self.call_number} members in {elapsed:.2f} seconds.')

        member = tarfile.data_filter(member, path)
        if member is None:
            return None

        if not ('images' or 'sparse' in member.name):
            return None

        if (match := path_re.match(member.name)) is None:
            return None

        scene_id = match.group(1)

        if scene_id in self.undistorted_scenes:
            return None

        if scene_id != self.current_scene:
            if self.current_scene is not None:
                try:
                    self.process_scene_callback(self.current_scene)
                except Exception as e:
                    print(f'Failed to process scene {self.current_scene}: {e}')
            self.current_scene = scene_id

        print(f'Allowing {member.name}.', end='\r')
        return member
    
    def process_scene_callback(self, scene_id: str):
        print(f'Processing scene {scene_id}.')
        distorted_dir = f'{unzip_path}/MegaDepth_v1_SfM/{scene_id}'
        sparse_dir = f'{distorted_dir}/sparse/manhattan'
        image_dir = f'{distorted_dir}/images'

        model_ids = os.listdir(sparse_dir)

        for model_id in model_ids:
            s_dir = f'{sparse_dir}/{model_id}'
            i_dir = image_dir
            d_dir = f'{undistort_path}/{scene_id}/{model_id}'
            self.undistort_model(s_dir, i_dir, d_dir)
        
        shutil.rmtree(distorted_dir)
    
    def undistort_model(self, sparse_dir: str, image_dir: str, dst_dir: str):
        os.makedirs(dst_dir, exist_ok=True)

        cmds = [
            COLMAP_EXE, 'image_undistorter',
            '--image_path', image_dir,
            '--input_path', sparse_dir,
            '--output_path', dst_dir,
            '--output_type', 'COLMAP',
            '--max_image_size', '1024',
        ]

        subprocess.run(cmds, capture_output=True)


filter = Filter()
with tarfile.open('/cvlabdata2/cvlab/datasets_tyszkiewicz/megadepth/MegaDepth_SfM_v1.tar.xz', mode='r:xz') as archive:
    archive.extractall(path=unzip_path, filter=filter)

# deal with the last scene
filter.process_scene_callback(filter.current_scene)