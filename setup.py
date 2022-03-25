from setuptools import setup

setup(
    name='disk',
    version='0.2.0',
    description='DISK local features. Paper: https://proceedings.neurips.cc/paper/2020/file/a42a596fc71e17828440030074d15e74-Paper.pdf',
    packages=['disk'],
    author='Michał Tyszkiewicz',
    author_email='michal.tyszkiewicz@epfl.ch',
    install_requires=[
        'torch',
        'imageio',
        'h5py',
        'opencv-python',
        'pydegensac',
        'tensorboard',
        'tensorflow',
        'tqdm',
        'unets @ git+https://github.com/jatentaki/unets.git',
        'torch-localize @ git+https://github.com/jatentaki/torch-localize.git',
        'torch-dimcheck @ git+https://github.com/jatentaki/torch-dimcheck.git@v2',
    ],
)
