
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import AssociativeRecallDataset
from src.model import AssociativeRecallTransformer
import os

save_dir = "/content/drive/MyDrive/transformer-copying-circuits"
os.makedirs(save_dir, exist_ok=True)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for tokens, targets in loader:
        tokens = tokens.to(device).long()
        targets = targets.to(device).long()

        with torch.set_grad_enabled(training):
            logits = model(tokens)

            last_logits = logits[:, -1, :]

            loss = F.cross_entropy(last_logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            last_logits.argmax(dim=-1) == targets
        ).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def main(args):
    best_test_acc = 0.0
    best_test_loss = float("inf")
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    vocab_size = args.n_keys + args.n_values
    seq_len = 2 * args.n_pairs + 1

    train_dataset = AssociativeRecallDataset(
        n_examples=args.n_train,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )

    test_dataset = AssociativeRecallDataset(
        n_examples=args.n_test,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_pairs=args.n_pairs,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size
    )

    model = AssociativeRecallTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss_before, train_acc_before = run_epoch(
            model, train_loader, device, optimizer
        )
        train_loss, train_acc = run_epoch(
                    model, train_loader, device
                )
        test_loss, test_acc = run_epoch(
            model, test_loader, device
        )
        if (
            test_acc > best_test_acc
            or (
                test_acc == best_test_acc
                and test_loss < best_test_loss
            )
        ):
            best_test_acc = test_acc
            best_test_loss = test_loss

            path = f"{save_dir}/model_{args.num_layers}layers.pt"

            torch.save(model.state_dict(), path)

            print(f"Saved model to {path}")
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.2%} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_acc:.2%}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
