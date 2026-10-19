import numpy as np
import pytest
import torch
from scipy.io import savemat
from that.model import THAT, pivotal_attention_mask, PTSG
from that.metrics import metrics
from that.cli.prepare_data import wald_pairs
from that.data import MatDataset, require_disjoint


def test_batch_and_head_independence():
    x = torch.tensor([[[[-2.0, -1.0], [0.0, 1.0]]]])
    y = x + 100
    expected = pivotal_attention_mask(x)
    assert torch.equal(expected, pivotal_attention_mask(torch.cat([x, y]))[:1])
    assert torch.equal(
        expected, pivotal_attention_mask(torch.cat([x, y], dim=1))[:, :1]
    )


def test_figure_three_routing_and_no_unused_parameters():
    model = THAT(5, 4)
    assert not any(
        "overlap" in n or "OCAB" in type(m).__name__ for n, m in model.named_modules()
    )
    seen = {}
    block = model.backbone.layers[0]
    handles = [
        block.ptsa.register_forward_hook(lambda m, i, o: seen.update(ptsa=o.detach())),
        block.wsa.register_forward_hook(lambda m, i, o: seen.update(wsa=o.detach())),
        block.mvfn.register_forward_pre_hook(
            lambda m, i: seen.update(mvfn_input=i[0].detach())
        ),
    ]
    output = model(torch.rand(2, 5, 4, 4), torch.rand(2, 1, 16, 16))
    torch.testing.assert_close(seen["mvfn_input"], seen["ptsa"] + seen["wsa"])
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    for h in handles:
        h.remove()
    assert any(isinstance(m, torch.nn.PixelShuffle) for m in model.modules())


def test_reference_metric_values():
    rng = np.random.RandomState(0)
    g = rng.rand(16, 16, 5) * 0.7 + 0.1
    r = metrics(g, g, 4)
    assert r["PSNR"] == float("inf") and r["ERGAS"] == 0
    assert r["SSIM"] == pytest.approx(1) and r["SAM"] == pytest.approx(0, abs=1e-6)
    assert r["SCC"] == pytest.approx(1)
    assert metrics(g + 0.1, g, 4)["PSNR"] == pytest.approx(20)
    # Independent closed form: reference 0.5, prediction 0.25.
    r = metrics(np.full((16, 16, 3), 0.25), np.full((16, 16, 3), 0.5), 4)
    assert r["ERGAS"] == pytest.approx(12.5)
    assert metrics(g + 0.1, g, 2)["ERGAS"] == pytest.approx(
        2 * metrics(g + 0.1, g, 4)["ERGAS"]
    )
    g = np.zeros((16, 16, 2))
    g[:, :, 0] = 1
    p = np.zeros_like(g)
    p[:, :, 1] = 1
    assert metrics(p, g, 4)["SAM"] == pytest.approx(90)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_wald_protocol(factor):
    cube = np.random.RandomState(0).rand(32, 40, 5)
    pair = wald_pairs(cube, factor, [0, 1], sigma=1.7, crop_size=16)
    assert pair["LRMS"].shape == (16 // factor, 16 // factor, 5)
    normalized = (cube - cube.min()) / (cube.max() - cube.min())
    np.testing.assert_allclose(pair["HRMS"], normalized[8:24, 12:28], atol=1e-7)
    np.testing.assert_allclose(pair["PAN"], pair["HRMS"][:, :, :2].mean(-1), atol=1e-7)
    for v in pair.values():
        assert np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1


def test_split_leakage(tmp_path):
    pair = wald_pairs(
        np.random.RandomState(2).rand(16, 16, 5), 4, [0], 1.0, crop_size=16
    )
    for name in ["train", "val"]:
        (tmp_path / name).mkdir()
        savemat(tmp_path / name / "sample.mat", pair)
    with pytest.raises(ValueError, match="leakage"):
        require_disjoint(MatDataset(tmp_path / "train"), MatDataset(tmp_path / "val"))


def test_gaussian_blurs_before_decimation():
    cube = np.zeros((16, 16, 3))
    cube[8, 8, :] = 1
    pair = wald_pairs(cube, 2, [0, 1], sigma=1.5, crop_size=16)
    # Direct subsampling has one nonzero pixel; Gaussian preprocessing spreads it.
    assert np.count_nonzero(pair["LRMS"][:, :, 0]) > 1
    assert pair["LRMS"].max() < pair["HRMS"].max()
