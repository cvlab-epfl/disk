import os
from typing import Callable, Any

import torch
import h5py
import numpy as np
import imageio
import lightning.pytorch as pl
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    Sampler,
    WeightedRandomSampler,
)
from torch import Tensor
from tqdm.auto import tqdm

from disk.common.image import Image
from disk.data.disk_dataset import DISKDataset

Transform = Callable[[Image], Image]

RANDOM_CROP_TRANSFORM: Transform = lambda image: image.random_crop((768, 768))
CENTER_CROP_TRANSFORM: Transform = lambda image: image.center_crop((768, 768))
DEFAULT_TRANSFORM: Transform = RANDOM_CROP_TRANSFORM


SceneFilter = Callable[[str, str], bool]
NO_SCENE_FILTER: SceneFilter = lambda _scene, _model: True

PairFilter = Callable[
    [
        np.ndarray,
    ],
    np.ndarray,
]  # int -> bool
NO_PAIR_FILTER: PairFilter = lambda n_matches: np.ones_like(n_matches, dtype=bool)


class ColmapModel(Dataset):
    def __init__(
        self,
        root: str,
        transform: Transform = DEFAULT_TRANSFORM,
        pair_filter: PairFilter = NO_PAIR_FILTER,
        subsample_n: None | int = None,
    ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.pair_filter = pair_filter

        with h5py.File(os.path.join(root, "metadata.h5"), "r") as metadata_file:
            self.image_names: list[str] = [
                name.decode("utf-8")
                for name in metadata_file["image_name"][()].tolist()
            ]
            self.R = torch.from_numpy(metadata_file["R"][()]).float()
            self.t = torch.from_numpy(metadata_file["t"][()]).float()
            self.K = torch.from_numpy(metadata_file["K"][()]).float()

            pairs: np.ndarray = np.asarray(metadata_file["pairs"][()])

            # filter pairs by number of matches
            overlaps: np.ndarray = np.asarray(metadata_file["overlaps"][()])
            pairs = pairs[self.pair_filter(overlaps)]

            # potentially subsample pairs
            if subsample_n is not None:
                generator = torch.Generator()
                generator.manual_seed(42)
                indices = torch.randperm(len(pairs), generator=generator)[:subsample_n]
                pairs = pairs[indices]

            self.pairs = np.ascontiguousarray(pairs)

    def __repr__(self):
        return f"ColmapModel(root={self.root}, len={len(self)}, transform={self.transform}, pair_filter={self.pair_filter})"

    def __len__(self):
        return len(self.pairs)

    @property
    def n_images(self) -> int:
        return len(self.image_names)

    def _get_bitmap_path(self, id: int) -> str:
        image_name = self.image_names[id]
        return os.path.join(self.root, "images", image_name)

    def _get_bitmap(self, id: int) -> Tensor:
        bitmap_path = self._get_bitmap_path(id)
        bitmap = imageio.imread(bitmap_path)
        bitmap = torch.from_numpy(bitmap)
        if len(bitmap.shape) == 2:  # grayscale
            bitmap = bitmap.unsqueeze(2).repeat(1, 1, 3)
        return bitmap.permute(2, 0, 1).float() / 255.0

    def get_image_by_id(self, id: int, transform: bool = True) -> Image:
        bitmap_path = self._get_bitmap_path(id)
        bitmap = self._get_bitmap(id)
        R, t, K = self.R[id], self.t[id], self.K[id]

        image = Image(K, R, t, bitmap, depth=None, bitmap_path=bitmap_path)

        if transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int) -> tuple[Image, Image]:
        id1, id2 = self.pairs[idx]

        image1 = self.get_image_by_id(id1)
        image2 = self.get_image_by_id(id2)

        return image1, image2


class AdjustedSampler(Sampler[int]):
    def __init__(self, dataset: ConcatDataset):
        self.subset_weights = [d.n_images for d in dataset.datasets]
        self.weighted_sampler = WeightedRandomSampler(
            self.subset_weights, replacement=True, num_samples=len(dataset)
        )
        self.subset_lengths = torch.tensor(
            [len(d) for d in dataset.datasets], dtype=torch.int64
        )
        self.subset_cumsums = torch.cumsum(self.subset_lengths, dim=0)

    def __iter__(self):
        for dataset_id in self.weighted_sampler:
            dataset_length = self.subset_lengths[dataset_id]
            dataset_offset = self.subset_cumsums[dataset_id] - dataset_length

            yield dataset_offset + torch.randint(dataset_length, size=(1,)).item()

    def __len__(self):
        return len(self.weighted_sampler)

class FixedSampler(Sampler[int]):
    def __init__(self, dataset: Dataset):
        generator = torch.Generator()
        generator.manual_seed(42)
        self.permutation = torch.randperm(len(dataset)).tolist()
    
    def __iter__(self):
        return iter(self.permutation)
    
    def __len__(self):
        return len(self.permutation)


class ColmapDataset(ConcatDataset):
    collate_fn = DISKDataset.collate_fn

    def __init__(
        self,
        root: str,
        filter: SceneFilter = NO_SCENE_FILTER,
        tiny_debug: bool = False,
        model_kwargs: dict[str, Any] = dict(),
    ):
        models = []
        for scene_name in tqdm(os.listdir(root)):
            if tiny_debug and len(models) > 5:
                break

            scene_path = os.path.join(root, scene_name)
            if not os.path.isdir(scene_path):
                continue

            for model_id in os.listdir(scene_path):
                model_path = os.path.join(scene_path, model_id)
                if not os.path.isdir(model_path):
                    continue

                if not filter(scene_name, model_id):
                    continue

                try:
                    model = ColmapModel(model_path, **model_kwargs)
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    continue

                if len(model) == 0:
                    continue

                models.append(model)

        super().__init__(models)


class ColmapDataModule(pl.LightningDataModule):
    train_filter = NO_SCENE_FILTER
    val_filter = NO_SCENE_FILTER
    test_filter = NO_SCENE_FILTER

    VAL_DATALOADER_NAMES = [
        "easy_val_dataloader",
        "medium_val_dataloader",
        "hard_val_dataloader",
    ]

    DEFAULT_LOADER_KWARGS = dict(
        batch_size=4,
        num_workers=8,
        collate_fn=ColmapDataset.collate_fn,
    )

    def __init__(
        self,
        root: str,
        loader_kwargs: dict[str, Any] = dict(),
        tiny_debug: bool = False,
        easy_training: bool = False,
    ):
        super().__init__()
        self.root = root
        self.loader_kwargs = {**self.DEFAULT_LOADER_KWARGS, **loader_kwargs}
        self.tiny_debug = tiny_debug
        self.easy_training = easy_training

    def get_split_scenes(self, split: str) -> list[str]:
        with open(os.path.join(self.root, f"{split}.txt"), "r") as f:
            return [line.strip() for line in f.readlines()]

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            if self.easy_training:
                pair_filter = lambda n_matches: n_matches >= 20
            else:
                pair_filter = NO_PAIR_FILTER

            self.train_dataset = ColmapDataset(
                self.root,
                filter=self.train_filter,
                tiny_debug=self.tiny_debug,
                model_kwargs=dict(
                    transform=RANDOM_CROP_TRANSFORM,
                    pair_filter=pair_filter,
                ),
            )

            self.easy_val_dataset = ColmapDataset(
                self.root,
                filter=self.val_filter,
                tiny_debug=self.tiny_debug,
                model_kwargs=dict(
                    subsample_n=128,
                    transform=CENTER_CROP_TRANSFORM,
                    pair_filter=lambda n_matches: n_matches >= 50,
                ),
            )

            self.medium_val_dataset = ColmapDataset(
                self.root,
                filter=self.val_filter,
                tiny_debug=self.tiny_debug,
                model_kwargs=dict(
                    subsample_n=128,
                    transform=CENTER_CROP_TRANSFORM,
                    pair_filter=lambda n_matches: (n_matches >= 10) & (n_matches < 50),
                ),
            )

            self.hard_val_dataset = ColmapDataset(
                self.root,
                filter=self.val_filter,
                tiny_debug=self.tiny_debug,
                model_kwargs=dict(
                    subsample_n=128,
                    transform=CENTER_CROP_TRANSFORM,
                    pair_filter=lambda n_matches: n_matches < 10,
                ),
            )

        elif stage == "test":
            self.test_dataset = ColmapDataset(
                self.root,
                filter=self.test_filter,
                transform=CENTER_CROP_TRANSFORM,
                tiny_debug=self.tiny_debug,
                subsample=128,
            )
        else:
            raise NotImplementedError(f"Unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=AdjustedSampler(self.train_dataset),
            **self.loader_kwargs,
        )

    def val_dataloader(self):
        return [
            DataLoader(self.easy_val_dataset, sampler=FixedSampler(self.easy_val_dataset), **self.loader_kwargs),
            DataLoader(self.medium_val_dataset, sampler=FixedSampler(self.medium_val_dataset), **self.loader_kwargs),
            DataLoader(self.hard_val_dataset, sampler=FixedSampler(self.hard_val_dataset), **self.loader_kwargs),
        ]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)


class MegadepthDataModule(ColmapDataModule):
    # fmt: off
    train_scenes = frozenset(['0217', '0472', '5008', '0231', '0057', '0285', '0076', '0360', '5002', '0027', '0559', '0733', '5006', '0455', '0104', '0141', '0258', '0238', '0303', '0269', '0067', '0148', '5000', '0252', '0017', '0037', '0034', '0186', '0224', '0389', '0205', '0115', '0478', '0380', '0043', '0189', '0327', '0160', '0064', '0181', '0156', '0046', '0323', '5012', '0060', '5011', '0286', '0496', '0200', '0406', '5001', '0446', '0149', '0049', '0402', '0183', '0001', '0041', '0271', '0257', '0150', '0197', '0098', '0100', '0099', '0326', '0407', '0004', '0190', '0235', '5007', '0394', '0090', '5010', '0214', '0482', '0306', '0062', '0101', '0147', '0243', '0240', '0087', '0102', '0117', '0005', '0042', '0058', '0335', '0036', '0151', '0505', '0341', '0237', '0130', '0080', '0162', '0493', '3346', '0023', '0035', '0007', '0277', '0185', '0137', '0411', '5009', '0331', '0095', '0387', '5005', '0039', '0012', '0275', '0071', '0281', '0299', '0061', '0070', '0312', '0122', '5003', '0348', '1017', '0212', '0083', '0056', '0377', '0086', '0065', '0476', '5004', '0307', '0044', '0290'])

    val_scenes = frozenset(['0430', '0015', '0229', '0033', '0025'])

    test_scenes = frozenset(['0412', '0019', '0008', '0129', '0349', '0016', '0078', '0092', '0024', '0032', '0021', '0022', '0002', '0107', '1589', '0003'])

    # fmt: on

    train_filter = lambda self, scene, model: scene in self.train_scenes
    val_filter = lambda self, scene, model: scene in self.val_scenes
    test_filter = lambda self, scene, model: scene in self.test_scenes
