import torch
from torch import Tensor

SQRT_2 = 1.414213


def distance_matrix(fs1: Tensor, fs2: Tensor) -> Tensor:
    """
    Assumes fs1 and fs2 are normalized!
    """
    return SQRT_2 * (1.0 - fs1 @ fs2.T).clamp(min=1e-6).sqrt()
