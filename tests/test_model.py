
import torch
import torch.nn.functional as F

from src.model import AssociativeRecallTransformer


def make_model(num_layers=2, seq_len=17):
    return AssociativeRecallTransformer(
        vocab_size=64,
        seq_len=seq_len,
        d_model=32,
        num_heads=2,
        num_layers=num_layers,
    )


def test_output_shape():
    model = make_model()
    x = torch.randint(0, 64, (4, 17))

    logits = model(x)

    assert logits.shape == (4, 17, 64)
    assert torch.isfinite(logits).all()


def test_causality():
    torch.manual_seed(42)
    model = make_model(seq_len=5)
    model.eval()

    x1 = torch.tensor([[3, 35, 7, 42, 3]])
    x2 = torch.tensor([[3, 35, 7, 42, 9]])

    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)

    assert torch.allclose(
        logits1[:, :-1, :],
        logits2[:, :-1, :],
        atol=1e-6,
    )


def test_gradient_flow():
    model = make_model()

    x = torch.randint(0, 64, (4, 17))
    targets = torch.randint(32, 64, (4,))

    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], targets)

    loss.backward()

    assert model.token_projection.weight.grad is not None
    assert model.blocks[0].W_q.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_num_layers():
    for num_layers in [1, 2, 3]:
        model = make_model(num_layers=num_layers)

        assert len(model.blocks) == num_layers

        x = torch.randint(0, 64, (4, 17))
        logits = model(x)

        assert logits.shape == (4, 17, 64)
