import torch, pydegensac, cv2, typing
from torch import Tensor

from disk import EstimationFailedError
from disk.geom import Pose


def _recover_pose(E: Tensor, i_coords: Tensor, j_coords: Tensor) -> Pose:
    E_ = E.to(torch.float64).numpy()
    i_coords_ = i_coords.to(torch.float64).numpy()
    j_coords_ = j_coords.to(torch.float64).numpy()

    # this function seems to only take f64 arguments, that's why
    # all the casting above
    n_inliers, R, T, inlier_mask = cv2.recoverPose(
        E_,
        i_coords_,
        j_coords_,
    )

    R = torch.from_numpy(R).to(torch.float32)
    # T is returned as a 3x1 array for some reason
    T = torch.from_numpy(T).to(torch.float32).squeeze(1)

    return Pose(R, T)


def _normalize_coords(coords: Tensor, K: Tensor) -> Tensor:
    coords = coords.to(torch.float32)

    f = torch.tensor([[K[0, 0], K[1, 1]]])
    c = torch.tensor([[K[0, 2], K[1, 2]]])

    return (coords - c) / f


class Ransac(typing.NamedTuple):
    reprojection_threshold: float = 1.0
    confidence: float = 0.9999
    max_iters: int = 10_000
    candidate_threshold: int = 10

    def __call__(
        self, left: Tensor, right: Tensor, K1: Tensor, K2: Tensor
    ) -> tuple[Pose, Tensor]:
        left = left.cpu()
        right = right.cpu()
        K1 = K1.cpu()
        K2 = K2.cpu()

        if left.shape[0] < self.candidate_threshold:
            raise EstimationFailedError()

        F, mask = pydegensac.findFundamentalMatrix(
            left.numpy(),
            right.numpy(),
            px_th=self.reprojection_threshold,
            conf=self.confidence,
            max_iters=self.max_iters,
        )

        # FIXME: how does pydegensac handle failure?
        if mask is None:
            raise EstimationFailedError()

        mask = torch.from_numpy(mask)
        F = torch.from_numpy(F).to(torch.float32)

        E = K2.T @ F @ K1

        try:
            pose = _recover_pose(
                E,
                _normalize_coords(left[mask], K1),
                _normalize_coords(right[mask], K2),
            )
        except cv2.error:
            raise EstimationFailedError()

        return pose, mask
