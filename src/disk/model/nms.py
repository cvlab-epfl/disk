import torch
import torch.nn.functional as F
from torch import Tensor


def nms(signal: Tensor, window_size: int = 5, cutoff: float = 0.0) -> Tensor:
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")

    _, ixs = F.max_pool2d(
        signal,
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2,
        return_indices=True,
    )

    # FIXME UPSTREAM: a workaround wrong shape of `ixs` until
    # https://github.com/pytorch/pytorch/issues/38986
    # is fixed
    if len(ixs.shape) == 4:
        assert ixs.shape[0] == 1
        ixs = ixs.squeeze(0)

    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)
