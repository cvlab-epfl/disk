from typing import Any

import torch
import lightning.pytorch as pl
from kornia.feature.disk import DISKFeatures
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from disk.model import DISK, ConsistentMatcher, CycleMatcher
from disk.loss import (
    Reinforce,
    PoseQuality,
    DiscreteMetric,
)
from disk.common.vis_kornia import visualize


class DiskLearner(pl.LightningModule):
    def __init__(self, disk: DISK, reward_class: type):
        super().__init__()

        self.automatic_optimization = False

        self.disk = disk
        self.reward_class = reward_class
        self.valtime_matcher = CycleMatcher()
        self.pose_quality_metric = PoseQuality()
        self.disc_quality_metric = DiscreteMetric(th=1.5, lm_kp=-0.01)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        bitmaps, images = batch
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])
        features_ = self.disk.features(bitmaps_, kind="rng")
        features = features_.reshape(*bitmaps.shape[:2])

        self.optimizers().zero_grad()
        stats = self.manual_backward(images, features, self.matcher)
        self.optimizers().step()

        for s in stats.flat:
            for k, v in s.items():
                self.log(f"train/{k}", v)

    def manual_backward(self, images, features, matcher):
        return self.loss_fn.accumulate_grad(images, features, matcher)

    def kornia_forward(self, images: Tensor) -> list[DISKFeatures]:
        old_features = self.disk.features(images, kind="nms")
        return [DISKFeatures(f.kp, f.desc, f.kp_logp) for f in old_features.flat]

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        bitmaps, images = batch
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])

        # at validation we use NMS extraction...
        features_ = self.disk.features(bitmaps_, kind="nms")
        features = features_.reshape(*bitmaps.shape[:2])

        # ...and nearest-neighbor matching
        matches = self.valtime_matcher.match_pairwise(features)
        d_stats = self.disc_quality_metric(images, matches)
        p_stats = self.pose_quality_metric(images, matches)

        for d_stat in d_stats.flat:
            for k, v in d_stat.items():
                self.log(f"val/hardness-{dataloader_idx}/{k}", v)

        for p_stat in p_stats.flat:
            for k, v in p_stat.items():
                self.log(f"val/hardness-{dataloader_idx}/{k}", v)

        if batch_idx == 0:
            for i, (imgs, feats) in enumerate(zip(images, features)):
                fig = visualize(feats, imgs)
                self.logger.experiment.add_figure(
                    f"val/hardness-{dataloader_idx}/vis-{i}",
                    fig,
                    global_step=self.global_step,
                )

    # def ramp(self, step: int) -> float:
    #    return max(0.0, min(1.0, (step - 250) / 100_000))

    # def on_train_batch_start(self, *args, **kwargs) -> None:
    #    ramp = self.ramp(self.global_step)

    #    self.loss_fn = Reinforce(
    #        self.reward_class(
    #            lm_tp=1.0,
    #            lm_fp=-0.25 * ramp,
    #            th=1.5,
    #        ),
    #        lm_kp=-0.001 * ramp,
    #    )

    #    # this is a module which is used to perform matching. It has a single
    #    # parameter called θ_M in the paper and `inverse_T` here. It could be
    #    # learned but I instead anneal it between 15 and 50
    #    inverse_T = 15 + 35 * ramp
    #    self.matcher = ConsistentMatcher(inverse_T=inverse_T).to(self.device)
    #    self.matcher.requires_grad_(False)

    def on_train_batch_start(self, *args, **kwargs) -> None:
        step = self.global_step

        if step == 0:
            self._adjust_loss(0)
        elif step == 250:
            self._adjust_loss(1)
        elif step >= 5250:
            step_adj = step - 250
            is_step = step_adj % 5000 == 0
            if is_step:
                e = (step_adj // 5000) + 1
                self._adjust_loss(e)

    def _adjust_loss(self, e: int) -> None:
        print(f"Adjusting loss for epoch {e}")
        if e == 0:
            ramp = 0.0
        elif e == 1:
            ramp = 0.1
        else:
            ramp = min(1.0, 0.1 + 0.2 * e)

        self.loss_fn = Reinforce(
            self.reward_class(
                lm_tp=1.0,
                lm_fp=-0.25 * ramp,
                th=1.5,
            ),
            lm_kp=-0.001 * ramp,
        )

        # this is a module which is used to perform matching. It has a single
        # parameter called θ_M in the paper and `inverse_T` here. It could be
        # learned but I instead anneal it between 15 and 50
        inverse_T = 15 + 35 * min(1.0, 0.05 * e)
        self.matcher = ConsistentMatcher(inverse_T=inverse_T)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.disk.parameters(), lr=1e-4)
