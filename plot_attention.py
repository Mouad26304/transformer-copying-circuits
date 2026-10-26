
import argparse
from pathlib import Path

import torch

from src.utils import (
    load_model,
    get_examples,
    extract_attention,
    plot_attention,
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

    # Generate fixed probe
    tokens, targets = get_examples(config, seed=args.seed)

    # Extract attention
    predictions, attentions = extract_attention(
        model, tokens
    )

    print("Tokens:", tokens.tolist())
    print("Target:", targets.item())
    print("Prediction:", predictions.item())
    print("Correct:", predictions.item() == targets.item())

    # Save results
    output_dir = Path(args.output_dir) / f"L{args.num_layers}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "tokens": tokens,
            "targets": targets,
            "predictions": predictions,
            "attentions": attentions,
            "config": config,
        },
        output_dir / "attention.pt",
    )

    plot_attention(
        attentions,
        tokens,
        output_dir,
        n_keys=args.n_keys,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/attention",
    )

    main(parser.parse_args())
