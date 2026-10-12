
import torch
from torch.utils.data import Dataset


class AssociativeRecallDataset(Dataset):
    """Generate synthetic key-value retrieval examples."""

    def __init__(
        self,
        n_examples=10_000,
        n_keys=32,
        n_values=32,
        n_pairs=8,
        seed=0,
    ):
        if not 1 <= n_pairs <= min(n_keys, n_values):
            raise ValueError(
                "n_pairs must fit within both vocabularies."
            )

        self.n_examples = n_examples
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_pairs = n_pairs
        self.seed = seed

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        # Each index produces a reproducible example.
        g = torch.Generator().manual_seed(self.seed + idx)

        # Sample distinct keys and values.
        keys = torch.randperm(
            self.n_keys, generator=g
        )[:self.n_pairs]

        values = torch.randperm(
            self.n_values, generator=g
        )[:self.n_pairs] + self.n_keys

        # Shuffle the key-value associations together.# actually no need to
        #order = torch.randperm(self.n_pairs, generator=g)
        #keys = keys[order]
        #values = values[order]

        # Select one key to query.
        query_idx = torch.randint(
            self.n_pairs, (1,), generator=g
        ).item()

        query_key = keys[query_idx]
        target = values[query_idx]

        # Interleave keys and values, then append the query.
        tokens = torch.stack((keys, values), dim=1).flatten()
        tokens = torch.cat((tokens, query_key.view(1)))

        return tokens.long(), target.long()