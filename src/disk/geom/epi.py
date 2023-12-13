import torch
from torch import Tensor


def cross_product_matrix(v: Tensor) -> Tensor:
    """following
    en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    """

    return torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=v.dtype,
        device=v.device,
    )


def xy_to_xyw(xy: Tensor) -> Tensor:
    ones = torch.ones(1, xy.shape[1], device=xy.device, dtype=xy.dtype)
    return torch.cat([xy, ones], dim=0)


def ims2E(im1, im2) -> Tensor:
    R = im2.R @ im1.R.T
    T = im2.T - R @ im1.T
    return cross_product_matrix(T) @ R


def ims2F(im1, im2) -> Tensor:
    E = ims2E(im1, im2)
    return im2.K_inv.T @ E @ im1.K_inv


def symdimm(x1: Tensor, x2: Tensor, im1, im2) -> Tensor:
    x1n = im1.K_inv @ xy_to_xyw(x1)
    x2n = im2.K_inv @ xy_to_xyw(x2)

    E = ims2E(im1, im2)

    E_x1 = E @ x1n
    Et_x2 = E.T @ x2n
    x2_E_x1 = x2n.T @ E_x1

    n = lambda v: torch.norm(v, p=2, dim=0)

    n1 = 1 / n(E_x1[:2])[None, :]
    n2 = 1 / n(Et_x2[:2])[:, None]
    norm = n1 + n2
    dist = x2_E_x1.pow(2) * norm
    return dist.T


def asymmdist(x1: Tensor, x2: Tensor, F: Tensor) -> Tensor:
    """
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    """

    x1_h = xy_to_xyw(x1)
    x2_h = xy_to_xyw(x2)

    Ft_x2 = F.T @ x2_h
    norm = torch.norm(Ft_x2[:2], p=2, dim=0)
    dist = (Ft_x2 / norm).T @ x1_h
    return dist.T


def asymmdist_from_imgs(x1: Tensor, x2: Tensor, im1, im2) -> Tensor:
    F = ims2F(im1, im2)
    return asymmdist(x1, x2, F)


def p_asymmdist(x1: Tensor, x2: Tensor, F: Tensor) -> Tensor:
    """
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    """

    x1_h = xy_to_xyw(x1)
    x2_h = xy_to_xyw(x2)

    Ft_x2 = F.T @ x2_h
    norm = torch.norm(Ft_x2[:2], p=2, dim=0)
    Ft_x2_n = Ft_x2 / norm

    return torch.einsum("ca,ca->a", (Ft_x2_n, x1_h))


def p_asymmdist_from_imgs(x1: Tensor, x2: Tensor, im1, im2) -> Tensor:
    F = ims2F(im1, im2)
    return p_asymmdist(x1, x2, F)
