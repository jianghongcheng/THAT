import torch
import pytest
from that.model import THAT, MVFN, kmeans_attn, pivotal_attention_mask

torch.set_num_threads(2)


def test_mask_and_empty_row_and_gradient():
    scores = torch.tensor(
        [[[[-9.0, -8.0, -7.0], [5.0, 6.0, 7.0], [-8.0, -7.0, -6.0]]]],
        requires_grad=True,
    )
    keep = pivotal_attention_mask(scores)
    assert keep.any(-1).all()
    weights = kmeans_attn(scores).softmax(-1)
    assert torch.isfinite(weights).all()
    assert (weights[~keep] == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    (weights * torch.arange(3.0)).sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    assert (scores.grad[~keep] == 0).all()


def test_constant_and_invalid_attention():
    torch.testing.assert_close(
        kmeans_attn(torch.zeros(1, 1, 3, 3)).softmax(-1),
        torch.full((1, 1, 3, 3), 1 / 3),
    )
    with pytest.raises(ValueError):
        kmeans_attn(torch.full((1, 1, 3, 3), float("nan")))


def test_paper_width_mvfn():
    module = MVFN(180)
    x = torch.randn(2, 180, 8, 8, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_full_default_network(factor):
    model = THAT(5, factor)
    assert model.backbone.embed_dim == 180
    x = torch.rand(1, 5, 16 // factor, 16 // factor)
    pan = torch.rand(1, 1, 16, 16)
    y = model(x, pan)
    assert y.shape == (1, 5, 16, 16) and torch.isfinite(y).all()
    y.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.backbone.conv_first.weight.grad.abs().sum() > 0
