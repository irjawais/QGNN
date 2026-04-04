"""
Training script for QGNN with Quantum Entanglement Loss.

Supports node classification, graph classification, graph regression,
and multi-label classification across all benchmark datasets.

Usage:
    python train.py --dataset cora
    python train.py --dataset peptides-struct --alpha 0.15
    python train.py --dataset pascalvoc-sp --epochs 500

See --help for all options.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

from model import QGNN
from datasets import load_dataset, DATASET_CONFIGS


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN with QEL")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument("--data_root", type=str, default="./data")

    # Architecture
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # QEL hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="QEL weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Global correlation strength")
    parser.add_argument("--k", type=int, default=16, help="Top-k eigenvalues for entropy")

    # Training
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--device", type=str, default="auto")

    # Output
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Training / evaluation for full-batch (Cora, Citeseer)
# ---------------------------------------------------------------------------

def train_fullbatch(model, data, optimizer, task):
    model.train()
    optimizer.zero_grad()

    out, qel = model(data.x, data.edge_index)

    if task == "node_classification":
        loss_task = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    else:
        loss_task = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss = loss_task + qel
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_fullbatch(model, data, task, mask):
    model.eval()
    out, _ = model(data.x, data.edge_index)

    if task == "node_classification":
        pred = out[mask].argmax(dim=-1)
        correct = (pred == data.y[mask]).sum().item()
        total = mask.sum().item()
        return correct / total
    return 0.0


# ---------------------------------------------------------------------------
# Training / evaluation for batched graphs (PPI, LRGB)
# ---------------------------------------------------------------------------

def train_batched(model, loader, optimizer, task, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out, qel = model(batch.x, batch.edge_index, batch.batch)

        if task == "graph_classification":
            loss_task = F.cross_entropy(out, batch.y)
        elif task == "graph_regression":
            loss_task = F.l1_loss(out, batch.y.float())
        elif task == "multilabel":
            loss_task = F.binary_cross_entropy_with_logits(out, batch.y.float())
        elif task == "node_classification":
            loss_task = F.cross_entropy(out, batch.y)
        elif task == "link_prediction":
            loss_task = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.float())
        else:
            loss_task = F.cross_entropy(out, batch.y)

        loss = loss_task + qel
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_batched(model, loader, task, metric, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out, _ = model(batch.x, batch.edge_index, batch.batch)

        if task == "graph_regression":
            all_preds.append(out.cpu())
            all_labels.append(batch.y.cpu())
        elif task == "graph_classification":
            all_preds.append(out.cpu())
            all_labels.append(batch.y.cpu())
        elif task == "multilabel":
            all_preds.append(torch.sigmoid(out).cpu())
            all_labels.append(batch.y.cpu())
        elif task == "node_classification":
            all_preds.append(out.cpu())
            all_labels.append(batch.y.cpu())
        else:
            all_preds.append(out.cpu())
            all_labels.append(batch.y.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return compute_metric(preds, labels, task, metric)


def compute_metric(preds, labels, task, metric):
    """Compute evaluation metric."""
    if metric == "accuracy":
        pred_classes = preds.argmax(dim=-1)
        return (pred_classes == labels).float().mean().item()

    elif metric == "f1":
        pred_classes = preds.argmax(dim=-1)
        return f1_score(
            labels.numpy(), pred_classes.numpy(), average="macro", zero_division=0
        )

    elif metric == "micro_f1":
        pred_binary = (preds > 0.5).float()
        return f1_score(
            labels.numpy(), pred_binary.numpy(), average="micro", zero_division=0
        )

    elif metric == "ap":
        return average_precision_score(
            labels.numpy(), preds.detach().numpy(), average="macro"
        )

    elif metric == "mae":
        return F.l1_loss(preds, labels.float()).item()

    elif metric == "hits_at_10":
        # Simplified hits@10 computation
        return 0.0  # Requires specialized link prediction evaluation

    return 0.0


def run_single(args, dataset_info, device, run_idx):
    """Execute a single training run."""
    config = dataset_info["config"]
    task = config["task"]
    metric_name = config["metric"]
    max_epochs = args.epochs or config["max_epochs"]

    # Build model
    model = QGNN(
        in_channels=config["in_channels"],
        hidden_channels=args.hidden_channels,
        out_channels=config["out_channels"],
        num_layers=args.num_layers,
        alpha=args.alpha,
        beta=args.beta,
        dropout=args.dropout,
        k=args.k,
        task=task,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    is_fullbatch = dataset_info["train_loader"] is None
    higher_is_better = metric_name not in ("mae",)

    best_val = -float("inf") if higher_is_better else float("inf")
    best_test = 0.0
    patience_counter = 0

    if is_fullbatch:
        data = dataset_info["data"].to(device)

    for epoch in range(1, max_epochs + 1):
        # Train
        if is_fullbatch:
            loss = train_fullbatch(model, data, optimizer, task)
        else:
            loss = train_batched(
                model, dataset_info["train_loader"], optimizer, task, device
            )

        # Evaluate
        if is_fullbatch:
            val_metric = eval_fullbatch(model, data, task, data.val_mask)
            test_metric = eval_fullbatch(model, data, task, data.test_mask)
        else:
            val_metric = eval_batched(
                model, dataset_info["val_loader"], task, metric_name, device
            )
            test_metric = eval_batched(
                model, dataset_info["test_loader"], task, metric_name, device
            )

        # Early stopping
        improved = (
            val_metric > best_val if higher_is_better else val_metric < best_val
        )
        if improved:
            best_val = val_metric
            best_test = test_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if args.verbose and epoch % 20 == 0:
            print(
                f"  Run {run_idx} | Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | Val {metric_name}: {val_metric:.4f} | "
                f"Test {metric_name}: {test_metric:.4f}"
            )

        if patience_counter >= args.patience:
            if args.verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_test


def main():
    args = get_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"QEL alpha={args.alpha}, beta={args.beta}, k={args.k}")
    print(f"Architecture: {args.num_layers} layers, hidden_dim={args.hidden_channels}")
    print("-" * 60)

    # Load dataset
    dataset_info = load_dataset(args.dataset, root=args.data_root)
    config = dataset_info["config"]
    metric_name = config["metric"]

    results = []
    for run in range(1, args.runs + 1):
        set_seed(args.seed + run)
        t0 = time.time()

        test_result = run_single(args, dataset_info, device, run)
        elapsed = time.time() - t0

        results.append(test_result)
        print(f"Run {run:2d}/{args.runs} | Test {metric_name}: {test_result:.4f} | Time: {elapsed:.1f}s")

    results = np.array(results)
    print("=" * 60)
    print(
        f"Final: {metric_name} = {results.mean():.4f} +/- {results.std():.4f} "
        f"(over {args.runs} runs)"
    )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_path = os.path.join(args.save_dir, f"{args.dataset}_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Metric: {metric_name}\n")
        f.write(f"Mean: {results.mean():.4f}\n")
        f.write(f"Std: {results.std():.4f}\n")
        f.write(f"All runs: {results.tolist()}\n")
        f.write(f"Args: {vars(args)}\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
