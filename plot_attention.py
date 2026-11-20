"""Analyze a trained associative-recall transformer and save attention heatmaps.

Run from the project root, after mounting Google Drive in Colab:
    python plot_attention.py --num-layers 1
    python plot_attention.py --num-layers 2
"""

import argparse
from pathlib import Path

import torch

from src.utils import (
    compute_attention_metrics,
    compute_behavior_metrics,
    extract_attention,
    get_examples,
    load_model,
    plot_attention,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = load_model(checkpoint_path, device, config)

    tokens, targets = get_examples(
        config,
        n_examples=args.n_examples,
        seed=args.seed,
    )
    predictions, attentions, logits = extract_attention(model, tokens)

    behavior = compute_behavior_metrics(logits, tokens, targets)
    print("\nBehavioral metrics")
    print(f"Accuracy: {behavior['accuracy']:.2%}")
    print(f"In-context rate: {behavior['in_context_rate']:.2%}")
    print(f"Mean value margin: {behavior['mean_value_margin']:.4f}")

    attention_results = compute_attention_metrics(tokens, attentions)
    for stats in attention_results:
        print(f"\nLayer {stats['layer']}")
        for head in range(len(stats["previous_key"])):
            print(
                f"Head {head} | "
                f"Previous key: {stats['previous_key'][head]:.3f} | "
                f"Other keys: {stats['other_keys'][head]:.3f} | "
                f"Correct value: {stats['correct_value'][head]:.3f} | "
                f"Other values: {stats['other_values'][head]:.3f} | "
                f"Correct value share: {stats['correct_value_share'][head]:.2%}"
            )

    # Plot the first input example, while metrics summarize all examples.
    output_dir = Path(args.output_dir) / f"L{args.num_layers}"
    plot_attention(attentions, tokens, save_dir=output_dir, n_keys=args.n_keys)
    print(f"\nAttention heatmaps saved to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/attention")
    main(parser.parse_args())
