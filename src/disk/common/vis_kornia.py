import matplotlib.pyplot as plt
import kornia.feature as KF
import kornia as K
from disk.common.structs import Features
from disk.common.image import Image
from disk.geom.epi import p_asymmdist_from_imgs
from kornia_moons.viz import draw_LAF_matches


def visualize(features, images) -> plt.Figure:
    assert len(features) == 1
    assert len(images) == 1

    features = features[0]
    images = images[0]

    f1: Features = features[0]
    f2: Features = features[1]
    im1: Image = images[0]
    im2: Image = images[1]

    _dists, idxs = KF.match_smnn(f1.desc, f2.desc, 0.95)

    mkpts1 = f1.kp[idxs[:, 0]]
    mkpts2 = f1.kp[idxs[:, 1]]

    asymmdist = p_asymmdist_from_imgs(mkpts1.T, mkpts2.T, im1, im2)
    is_good = asymmdist < 5

    fig, ax = draw_LAF_matches(
        KF.laf_from_center_scale_ori(f1.kp[None].cpu()),
        KF.laf_from_center_scale_ori(f2.kp[None].cpu()),
        idxs.cpu(),
        K.tensor_to_image(im1.bitmap.cpu()),
        K.tensor_to_image(im2.bitmap.cpu()),
        is_good,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1, 1, 0.2, 0.3),
            "feature_color": None,
            "vertical": False,
        },
        return_fig_ax=True,
    )

    ax.axis("off")

    return fig
