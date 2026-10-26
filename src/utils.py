
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

    return predictions.cpu(), attentions


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
