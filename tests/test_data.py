
from src.data import AssociativeRecallDataset
import torch


def test_sequence_shape_and_types(
    n_examples=10_000,
    n_keys=32,
    n_values=32,
    n_pairs=8,
    seed=0,
):
    dataset = AssociativeRecallDataset(
        n_examples, n_keys, n_values, n_pairs, seed
    )

    # One example is enough to check shape and dtype.
    example, target = dataset[0]

    assert example.dtype == torch.long
    assert example.shape == (2 * n_pairs + 1,)
    assert target.dtype == torch.long
    assert target.ndim == 0


def test_target_matches_query(
    n_examples=100,
    n_keys=32,
    n_values=32,
    n_pairs=8,
    seed=0,
):
    dataset = AssociativeRecallDataset(
        n_examples, n_keys, n_values, n_pairs, seed
    )

    for example_idx in range(n_examples):
        example, target = dataset[example_idx]

        pairs = example[:-1].reshape(n_pairs, 2)
        query = example[-1]

        mapping = {
            key.item(): value.item()
            for key, value in pairs
        }
        assert query.item() in mapping
        assert mapping[query.item()] == target.item()


def test_same_index_is_reproducible():
    dataset = AssociativeRecallDataset(seed=42)

    example1, target1 = dataset[5]
    example2, target2 = dataset[5]

    assert torch.equal(example1, example2)
    assert torch.equal(target1, target2)


def test_token_ranges(
    n_examples=100,
    n_keys=32,
    n_values=32,
    n_pairs=8,
    seed=0,
):
    dataset = AssociativeRecallDataset(
        n_examples, n_keys, n_values, n_pairs, seed
    )

    for example_idx in range(n_examples):
        example, target = dataset[example_idx]

        pairs = example[:-1].reshape(n_pairs, 2)
        keys = pairs[:, 0]
        values = pairs[:, 1]
        query = example[-1]

        assert torch.all((keys >= 0) & (keys < n_keys))
        assert torch.all(
            (values >= n_keys) & (values < n_keys + n_values)
        )
        assert 0 <= query.item() < n_keys
        assert n_keys <= target.item() < n_keys + n_values

        # Keys and values should be distinct within each example.
        assert len(torch.unique(keys)) == n_pairs
        assert len(torch.unique(values)) == n_pairs