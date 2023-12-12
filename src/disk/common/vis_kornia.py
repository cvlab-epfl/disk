import matplotlib.pyplot as plt
import kornia.feature as KF
import kornia as K
from kornia_moons.viz import draw_LAF_matches
from disk.common.structs import Features
from disk.common.image import Image
from disk.geom.epi import p_asymmdist_from_imgs

def visualize(features: list[Features], images: list[Image], threshold_px: float = 2.5) -> plt.Figure:
    assert len(features) == 2
    assert len(images) == 2

    f1, f2 = features
    im1, im2 = images

    _dists, idxs = KF.match_smnn(f1.desc, f2.desc, 0.95)

    mkpts1 = f1.kp[idxs[:, 0]]
    mkpts2 = f2.kp[idxs[:, 1]]

    asymmdist_ab = p_asymmdist_from_imgs(mkpts1.T, mkpts2.T, im1, im2).abs()
    asymmdist_ba = p_asymmdist_from_imgs(mkpts2.T, mkpts1.T, im2, im1).abs()
    is_good = (asymmdist_ab < threshold_px) & (asymmdist_ba < threshold_px)

    fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(f1.kp[None].cpu()),
        KF.laf_from_center_scale_ori(f2.kp[None].cpu()),
        idxs.cpu(),
        K.tensor_to_image(im1.bitmap.cpu()),
        K.tensor_to_image(im2.bitmap.cpu()),
        is_good.cpu(),
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1, 1, 0.2, 0.3),
            "feature_color": None,
            "vertical": False,
        },
        fig=fig,
        ax=ax,
    )

    ax.axis("off")

    return fig
