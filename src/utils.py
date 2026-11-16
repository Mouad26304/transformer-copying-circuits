
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.model import AssociativeRecallTransformer
from src.data import AssociativeRecallDataset


def load_model(checkpoint_path, device, config):
    """Reconstruct a trained transformer and load its weights."""

    model = AssociativeRecallTransformer(
        vocab_size=config["n_keys"] + config["n_values"],
        seq_len=2 * config["n_pairs"] + 1,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def get_examples(config, n_examples=1, seed=1234):
    dataset = AssociativeRecallDataset(
        n_examples=n_examples,
        n_keys=config["n_keys"],
        n_values=config["n_values"],
        n_pairs=config["n_pairs"],
        seed=seed,
    )

    tokens = []
    targets = []

    for i in range(n_examples):
        x, y = dataset[i]
        tokens.append(torch.as_tensor(x).long())
        targets.append(torch.as_tensor(y).long())

    return torch.stack(tokens), torch.stack(targets)


@torch.no_grad()
def extract_attention(model, tokens):
    """Return predictions and attention matrices for each layer."""

    device = next(model.parameters()).device
    tokens = tokens.to(device)

    model.eval()

    for block in model.blocks:
        block.capture_attention = True
        block.last_attention = None

    try:
        logits = model(tokens)
        predictions = logits[:, -1, :].argmax(dim=-1)

        attentions = []

        for block in model.blocks:
            attn = block.last_attention

            if attn is None:
                raise RuntimeError(
                    "Attention was not captured. "
                    "Check GPT2Block.forward()."
                )

            attentions.append(attn.detach().cpu())

    finally:
        for block in model.blocks:
            block.capture_attention = False
            block.last_attention = None

    return predictions.cpu(), attentions, logits.detach().cpu()


def plot_attention(attentions, tokens, save_dir, n_keys):
    """Save one attention heatmap figure per transformer layer."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    token_ids = tokens[0].cpu().tolist()

    labels = []
    for i, token_id in enumerate(token_ids):
        if i == len(token_ids) - 1:
            labels.append(f"Q:{token_id}")
        elif i % 2 == 0:
            labels.append(f"K:{token_id}")
        else:
            labels.append(f"V:{token_id - n_keys}")

    for layer_idx, attention in enumerate(attentions):
        # Shape: (batch, heads, seq_len, seq_len)
        attention = attention[0].numpy()

        n_heads = attention.shape[0]

        fig, axes = plt.subplots(
            1, n_heads,
            figsize=(4 * n_heads, 4),
            squeeze=False,
        )

        for head_idx in range(n_heads):
            ax = axes[0, head_idx]

            im = ax.imshow(
                attention[head_idx],
                cmap="Blues",
                vmin=0,
                vmax=1,
            )

            ax.set_title(f"Layer {layer_idx}, Head {head_idx}")

            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")

        fig.tight_layout()

        fig.savefig(
            save_dir / f"layer_{layer_idx}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

def compute_behavior_metrics(logits, tokens, targets):
    """Measure recall, in-context prediction, and logit margin."""

    if logits.ndim == 3:
        logits = logits[:, -1, :]

    tokens = tokens.to(logits.device)
    targets = targets.reshape(-1).to(logits.device)

    # Values occupy positions 1, 3, 5, 7, ...
    values = tokens[:, 1:-1:2].long()

    if values.shape[1] < 2:
        raise ValueError("Need at least two key-value pairs.")

    if not torch.all((values == targets[:, None]).sum(-1) == 1):
        raise ValueError("Each target must appear exactly once among input values.")

    # 1. Exact recall accuracy
    predicted = logits.argmax(dim=-1)
    accuracy = (predicted == targets).float().mean()

    # 2. Prediction belongs to the values in context
    in_context = (predicted[:, None] == values).any(dim=-1)
    in_context_rate = in_context.float().mean()

    # 3. Margin against the strongest incorrect in-context value
    candidate_logits = logits.gather(dim=1, index=values)

    wrong_logits = candidate_logits.masked_fill(
        values == targets[:, None],
        float("-inf"),
    ).max(dim=-1).values

    correct_logits = logits.gather(
        dim=1, index=targets[:, None]
    ).squeeze(-1)

    margin = correct_logits - wrong_logits

    return {
        "accuracy": accuracy.item(),
        "in_context_rate": in_context_rate.item(),
        "mean_value_margin": margin.mean().item(),
    }


def compute_attention_metrics(tokens, attentions):
    """
    Measure attention patterns for each layer and head.

    attentions: list of tensors (B, H, T, T)
    tokens: (B, T)
    """

    tokens = tokens.cpu()
    n_pairs = (tokens.shape[1] - 1) // 2

    if n_pairs < 2:
        raise ValueError("Need at least two pairs.")

    # Keys: positions 0, 2, 4, ...
    keys = tokens[:, :-1:2]

    # Query: final position
    query = tokens[:, -1:]

    # Find which key matches the query
    matching = keys == query

    if not torch.all(matching.sum(dim=-1) == 1):
        raise ValueError("Expected exactly one matching key.")

    correct_pair_index = matching.long().argmax(dim=-1)

    results = []

    for layer, attention in enumerate(attentions):
        attention = attention.detach().cpu()

        if (
            attention.ndim != 4
            or attention.shape[0] != tokens.shape[0]
            or attention.shape[-2:]
            != (tokens.shape[1], tokens.shape[1])
        ):
            raise ValueError(
                f"Invalid attention shape in layer {layer}"
            )

        # -------------------------------------
        # 1. Query -> correct value
        # -------------------------------------

        # (B, H, n_pairs)
        value_attention = attention[:, :, -1, 1:-1:2]

        batch_size, n_heads, _ = value_attention.shape

        indices = correct_pair_index[:, None, None].expand(
            -1, n_heads, 1
        )

        # (B, H)
        correct_value = value_attention.gather(
            dim=2, index=indices
        ).squeeze(-1)

        other_values = (
            value_attention.sum(dim=-1) - correct_value
        ) / (n_pairs - 1)

        # Fraction of attention to values assigned to correct one
        correct_value_share = correct_value / (
            value_attention.sum(dim=-1).clamp_min(1e-12)
        )

        # -------------------------------------
        # 2. Value -> preceding key
        # -------------------------------------

        previous_keys = []
        other_keys = []

        # Skip first pair: it has no competing preceding keys
        for pair_idx in range(1, n_pairs):
            value_pos = 2 * pair_idx + 1
            correct_key_pos = 2 * pair_idx

            previous_keys.append(
                attention[:, :, value_pos, correct_key_pos]
            )

            other_key_positions = torch.arange(
                0, correct_key_pos, 2
            )

            other_keys.append(
                attention[
                    :, :, value_pos, other_key_positions
                ].mean(dim=-1)
            )

        # Average over examples and value positions
        previous_key_mean = torch.stack(
            previous_keys, dim=-1
        ).mean(dim=(0, 2))

        other_key_mean = torch.stack(
            other_keys, dim=-1
        ).mean(dim=(0, 2))

        results.append({
            "layer": layer,
            "previous_key": previous_key_mean,
            "other_keys": other_key_mean,
            "correct_value": correct_value.mean(dim=0),
            "other_values": other_values.mean(dim=0),
            "correct_value_share": correct_value_share.mean(dim=0),
        })

    return results
