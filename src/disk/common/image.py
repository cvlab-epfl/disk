from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch import Tensor


def _rescale(tensor: Tensor, size: int) -> Tensor:
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _pad(tensor: Tensor, size, value=0.0) -> Tensor:
    xpad = size[1] - tensor.shape[2]
    ypad = size[0] - tensor.shape[1]

    # not that F.pad takes sizes starting from the last dimension
    padded = F.pad(tensor, (0, xpad, 0, ypad), mode="constant", value=value)

    assert padded.shape[1:] == tuple(size)
    return padded


@dataclass
class Image:
    K: Tensor
    R: Tensor
    T: Tensor
    bitmap: Tensor
    depth: Tensor | None
    bitmap_path: str  # for debugging

    @property
    def K_inv(self):
        return self.K.inverse()

    @property
    def hwc(self):
        return self.bitmap.permute(1, 2, 0)

    @property
    def shape(self):
        return self.bitmap.shape[1:]

    def scale(self, size):
        """
        Rescale the image to at most size=(height, width). One dimension is
        guaranteed to be equally matched
        """

        x_factor = self.shape[0] / size[0]
        y_factor = self.shape[1] / size[1]

        f = 1 / max(x_factor, y_factor)
        if x_factor > y_factor:
            new_size = (size[0], int(f * self.shape[1]))
        else:
            new_size = (int(f * self.shape[0]), size[1])

        K_scaler = torch.tensor(
            [[f, 0, 0], [0, f, 0], [0, 0, 1]], dtype=self.K.dtype, device=self.K.device
        )
        K = K_scaler @ self.K

        bitmap = _rescale(self.bitmap, new_size)
        if self.depth is not None:
            depth = _rescale(self.depth, new_size)
        else:
            depth = None

        return Image(K, self.R, self.T, bitmap, depth, self.bitmap_path)

    def pad(self, size):
        bitmap = _pad(self.bitmap, size, value=0)
        if self.depth is not None:
            depth = _pad(self.depth, size, value=float("NaN"))
        else:
            depth = None

        return Image(self.K, self.R, self.T, bitmap, depth, self.bitmap_path)

    def crop(
        self, src_bbox: tuple[int, int, int, int], dst_size: tuple[int, int]
    ) -> Image:
        bitmap = self.bitmap.numpy()
        src_w_min, src_h_min, src_w_max, src_h_max = src_bbox
        dst_w_max, dst_h_max = dst_size

        src_square = np.array(
            [
                [src_h_min, src_w_min],
                [src_h_max, src_w_min],
                [src_h_min, src_w_max],
                [src_h_max, src_w_max],
            ],
            dtype=np.float32,
        )
        dst_square = np.array(
            [[0, 0], [dst_h_max, 0], [0, dst_w_max], [dst_h_max, dst_w_max]],
            dtype=np.float32,
        )

        warp_mat = cv2.getPerspectiveTransform(src_square, dst_square)

        new_bitmap = cv2.warpPerspective(
            bitmap.transpose(1, 2, 0), warp_mat, (dst_h_max, dst_w_max)
        )
        new_K = warp_mat.astype(np.float32) @ self.K.numpy()

        return Image(
            K=torch.from_numpy(new_K),
            R=self.R,
            T=self.T,
            bitmap=torch.from_numpy(new_bitmap).permute(2, 0, 1),
            depth=self.depth,
            bitmap_path=self.bitmap_path,
        )

    def center_crop(self, crop_size: tuple[int, int]) -> Image:
        crop_corner = np.array(crop_size)
        img_corner = np.array(self.bitmap.shape[1:])

        ratio = crop_corner / img_corner
        ratio = ratio.max()

        img_h, img_w = crop_corner / ratio

        delta_h = (img_corner[0] - img_h) // 2
        delta_w = (img_corner[1] - img_w) // 2

        return self.crop(
            (delta_h, delta_w, img_h + delta_h, img_w + delta_w), crop_size
        )

    def random_crop(self, crop_size: tuple[int, int]) -> Image:
        crop_corner = np.array(crop_size)
        img_corner = np.array(self.bitmap.shape[1:])

        ratio = crop_corner / img_corner
        ratio = ratio.max()

        img_h, img_w = crop_corner / ratio

        h_range = int(img_corner[0] - img_h)
        w_range = int(img_corner[1] - img_w)

        assert h_range >= 0 and w_range >= 0

        if h_range == 0:
            delta_h = 0
        else:
            delta_h = np.random.randint(0, h_range)

        if w_range == 0:
            delta_w = 0
        else:
            delta_w = np.random.randint(0, w_range)

        return self.crop(
            (delta_h, delta_w, img_h + delta_h, img_w + delta_w), crop_size
        )

    def to(self, *args, **kwargs):
        # use getattr/setattr to avoid repetitive code.
        # exclude `self.bitmap` because we don't need it on GPU (it's treated
        # separately by the dataloader)
        TRANSFERRED_ATTRS = ["K", "R", "T", "depth"]

        for key in TRANSFERRED_ATTRS:
            attr = getattr(self, key)
            if attr is not None:
                attr_transferred = attr.to(*args, **kwargs)
            setattr(self, key, attr_transferred)

        return self

    def unproject(self, xy: Tensor) -> Tensor:
        depth = self.fetch_depth(xy)

        xyw = torch.cat(
            [
                xy.to(depth.dtype),
                torch.ones(1, xy.shape[1], dtype=depth.dtype, device=xy.device),
            ],
            dim=0,
        )

        xyz = (self.K_inv @ xyw) * depth
        xyz_w = self.R.T @ (xyz - self.T[:, None])

        return xyz_w

    def project(self, xyw: Tensor) -> Tensor:
        extrinsic = self.R @ xyw + self.T[:, None]
        intrinsic = self.K @ extrinsic
        return intrinsic[:2] / intrinsic[2]

    def in_range_mask(self, xy: Tensor) -> Tensor:
        h, w = self.shape
        x, y = xy

        return (0 <= x) & (x < w) & (0 <= y) & (y < h)

    def fetch_depth(self, xy: Tensor) -> Tensor:
        if self.depth is None:
            raise ValueError(f"Depth is not loaded")

        in_range = self.in_range_mask(xy)
        finite = torch.isfinite(xy).all(dim=0)
        valid_depth = in_range & finite
        x, y = xy[:, valid_depth].to(torch.int64)
        depth = torch.full(
            (xy.shape[1],),
            fill_value=float("NaN"),
            device=xy.device,
            dtype=self.depth.dtype,
        )
        depth[valid_depth] = self.depth[0, y, x]

        return depth
