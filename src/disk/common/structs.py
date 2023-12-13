import abc
from dataclasses import dataclass
from torch import Tensor

NpArray = list  # type alias


@dataclass
class Features:
    kp: Tensor
    desc: Tensor
    kp_logp: Tensor | None

    def __post_init__(self):
        assert self.kp.device == self.desc.device
        assert self.kp.device == self.kp_logp.device

    @property
    def n(self):
        return self.kp.shape[0]

    @property
    def device(self):
        return self.kp.device

    def detached_and_grad_(self):
        return Features(
            self.kp,
            self.desc.detach().requires_grad_(),
            self.kp_logp.detach().requires_grad_(),
        )

    def requires_grad_(self, is_on):
        self.desc.requires_grad_(is_on)
        self.kp_logp.requires_grad_(is_on)

    def grad_tensors(self):
        return [self.desc, self.kp_logp]

    def to(self, *args, **kwargs):
        return Features(
            self.kp.to(*args, **kwargs),
            self.desc.to(*args, **kwargs),
            self.kp_logp.to(*args, **kwargs) if self.kp_logp is not None else None,
        )


class MatchDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> [2, "K"]:
        pass

    @abc.abstractmethod
    def mle(self) -> [2, "K"]:
        pass

    @abc.abstractmethod
    def dense_logp(self):
        pass

    @abc.abstractmethod
    def dense_p(self):
        pass

    @abc.abstractmethod
    def features_1(self) -> Features:
        pass

    @abc.abstractmethod
    def features_2(self) -> Features:
        pass

    @property
    def shape(self):
        return self.features_1().kp.shape[0], self.features_2().kp.shape[1]

    def matched_pairs(self, mle=False):
        matches = self.mle() if mle else self.sample()

        return MatchedPairs(
            self.features_1().kp,
            self.features_2().kp,
            matches,
        )


@dataclass
class MatchedPairs:
    kps1: Tensor
    kps2: Tensor
    matches: Tensor

    def to(self, *args, **kwargs):
        return MatchedPairs(
            self.kps1.to(*args, **kwargs),
            self.kps2.to(*args, **kwargs),
            self.matches.to(*args, **kwargs),
        )
