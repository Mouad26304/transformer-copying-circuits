
import argparse
from pathlib import Path

import torch

from src.utils import (
    load_model,
    get_examples,
    extract_attention,
    plot_attention,
)
from src.utils import (
    compute_behavior_metrics,
    compute_attention_metrics,
)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = {
        "num_layers": args.num_layers,
        "n_keys": args.n_keys,
        "n_values": args.n_values,
        "n_pairs": args.n_pairs,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
    }

    checkpoint_path = args.checkpoint or (
        "/content/drive/MyDrive/transformer-copying-circuits/"
        f"model_{args.num_layers}layers.pt"
    )

    # Load trained transformer
    model = load_model(checkpoint_path, device, config)

    
    # Generate new, reproducible examples
    tokens, targets = get_examples(
        config,
        n_examples=1000,
        seed=100000,
    )

    predictions, attentions, logits = extract_attention(
        model, tokens
    )

    # Behavioral metrics
    behavior = compute_behavior_metrics(
        logits, tokens, targets
    )

    print("\nBehavioral metrics")
    print(f"Accuracy: {behavior['accuracy']:.2%}")
    print(f"In-context rate: {behavior['in_context_rate']:.2%}")
    print(f"Mean value margin: {behavior['mean_value_margin']:.4f}")

    # Attention statistics
    attention_results = compute_attention_metrics(
        tokens, attentions
    )

    for stats in attention_results:
        print(f"\nLayer {stats['layer']}")

        n_heads = len(stats["previous_key"])

        for head in range(n_heads):
            print(
                f"Head {head} | "
                f"Previous key: {stats['previous_key'][head]:.3f} | "
                f"Other keys: {stats['other_keys'][head]:.3f} | "
                f"Correct value: {stats['correct_value'][head]:.3f} | "
                f"Other values: {stats['other_values'][head]:.3f} | "
                f"Correct value share: "
                f"{stats['correct_value_share'][head]:.2%}"
            )

