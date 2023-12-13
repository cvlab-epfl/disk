import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from disk import Features, NpArray, MatchDistribution
from disk.geom import distance_matrix


class ConsistentMatchDistribution(MatchDistribution):
    def __init__(
        self,
        features_1: Features,
        features_2: Features,
        inverse_T: float,
    ):
        self._features_1 = features_1
        self._features_2 = features_2
        self.inverse_T = inverse_T

        distances = distance_matrix(
            self.features_1().desc,
            self.features_2().desc,
        )
        affinity = -inverse_T * distances

        self._cat_I = Categorical(logits=affinity)
        self._cat_T = Categorical(logits=affinity.T)

        self._dense_logp = None
        self._dense_p = None

    def dense_p(self) -> Tensor:
        if self._dense_p is None:
            self._dense_p = self._cat_I.probs * self._cat_T.probs.T

        return self._dense_p

    def dense_logp(self) -> Tensor:
        if self._dense_logp is None:
            self._dense_logp = self._cat_I.logits + self._cat_T.logits.T

        return self._dense_logp

    def _select_cycle_consistent(self, left: Tensor, right: Tensor) -> Tensor:
        indexes = torch.arange(left.shape[0], device=left.device)
        cycle_consistent = right[left] == indexes

        paired_left = left[cycle_consistent]

        return torch.stack(
            [
                right[paired_left],
                paired_left,
            ],
            dim=0,
        )

    def sample(self) -> Tensor:
        samples_I = self._cat_I.sample()
        samples_T = self._cat_T.sample()

        return self._select_cycle_consistent(samples_I, samples_T)

    def mle(self) -> Tensor:
        maxes_I = self._cat_I.logits.argmax(dim=1)
        maxes_T = self._cat_T.logits.argmax(dim=1)

        # FIXME UPSTREAM: this detachment is necessary until the bug is fixed
        maxes_I = maxes_I.detach()
        maxes_T = maxes_T.detach()

        return self._select_cycle_consistent(maxes_I, maxes_T)

    def features_1(self) -> Features:
        return self._features_1

    def features_2(self) -> Features:
        return self._features_2


class ConsistentMatcher(torch.nn.Module):
    def __init__(self, inverse_T=1.0):
        super(ConsistentMatcher, self).__init__()
        self.inverse_T = nn.Parameter(torch.tensor(inverse_T, dtype=torch.float32))

    def extra_repr(self):
        return f"inverse_T={self.inverse_T.item()}"

    def match_pair(self, features_1: Features, features_2: Features):
        return ConsistentMatchDistribution(features_1, features_2, self.inverse_T)
