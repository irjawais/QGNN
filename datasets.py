"""
Dataset loading utilities for QGNN experiments.

Supports:
- Standard benchmarks: Cora, Citeseer, PPI
- Long Range Graph Benchmark (LRGB): PascalVOC-SP, COCO-SP, Peptides-func,
  Peptides-struct, PCQM-Contact
"""

import os

import torch
from torch_geometric.datasets import Planetoid, PPI, LRGBDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms import NormalizeFeatures


DATASET_CONFIGS = {
    # Standard benchmarks
    "cora": {
        "task": "node_classification",
        "metric": "accuracy",
        "num_classes": 7,
        "max_epochs": 200,
        "batch_size": None,  # Full-batch
    },
    "citeseer": {
        "task": "node_classification",
        "metric": "accuracy",
        "num_classes": 6,
        "max_epochs": 200,
        "batch_size": None,
    },
    "ppi": {
        "task": "multilabel",
        "metric": "micro_f1",
        "num_classes": 121,
        "max_epochs": 200,
        "batch_size": 2,
    },
    # LRGB benchmarks
    "pascalvoc-sp": {
        "task": "node_classification",
        "metric": "f1",
        "num_classes": 21,
        "max_epochs": 500,
        "batch_size": 32,
    },
    "coco-sp": {
        "task": "node_classification",
        "metric": "f1",
        "num_classes": 81,
        "max_epochs": 500,
        "batch_size": 32,
    },
    "peptides-func": {
        "task": "graph_classification",
        "metric": "ap",
        "num_classes": 10,
        "max_epochs": 500,
        "batch_size": 32,
    },
    "peptides-struct": {
        "task": "graph_regression",
        "metric": "mae",
        "num_classes": 11,
        "max_epochs": 500,
        "batch_size": 32,
    },
    "pcqm-contact": {
        "task": "link_prediction",
        "metric": "hits_at_10",
        "num_classes": 1,
        "max_epochs": 500,
        "batch_size": 32,
    },
}


def load_dataset(name: str, root: str = "./data"):
    """Load a dataset by name.

    Args:
        name: Dataset name (case-insensitive). One of: cora, citeseer, ppi,
              pascalvoc-sp, coco-sp, peptides-func, peptides-struct, pcqm-contact.
        root: Root directory for data storage.

    Returns:
        Dictionary with keys: train_loader, val_loader, test_loader, config, dataset.
    """
    name = name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[name]
    os.makedirs(root, exist_ok=True)

    if name in ("cora", "citeseer"):
        return _load_planetoid(name, root, config)
    elif name == "ppi":
        return _load_ppi(root, config)
    else:
        return _load_lrgb(name, root, config)


def _load_planetoid(name: str, root: str, config: dict) -> dict:
    """Load Cora or Citeseer with standard public splits."""
    dataset = Planetoid(
        root=os.path.join(root, "Planetoid"),
        name=name.capitalize(),
        transform=NormalizeFeatures(),
    )
    data = dataset[0]

    return {
        "data": data,
        "dataset": dataset,
        "train_loader": None,  # Full-batch; use data directly
        "val_loader": None,
        "test_loader": None,
        "config": {
            **config,
            "in_channels": dataset.num_node_features,
            "out_channels": dataset.num_classes,
        },
    }


def _load_ppi(root: str, config: dict) -> dict:
    """Load PPI dataset with standard 20/2/2 graph split."""
    train_dataset = PPI(root=os.path.join(root, "PPI"), split="train")
    val_dataset = PPI(root=os.path.join(root, "PPI"), split="val")
    test_dataset = PPI(root=os.path.join(root, "PPI"), split="test")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    return {
        "data": None,
        "dataset": train_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "config": {
            **config,
            "in_channels": train_dataset.num_node_features,
            "out_channels": train_dataset.num_classes,
        },
    }


def _load_lrgb(name: str, root: str, config: dict) -> dict:
    """Load an LRGB dataset with official train/val/test splits."""
    lrgb_root = os.path.join(root, "LRGB")

    train_dataset = LRGBDataset(root=lrgb_root, name=name, split="train")
    val_dataset = LRGBDataset(root=lrgb_root, name=name, split="val")
    test_dataset = LRGBDataset(root=lrgb_root, name=name, split="test")

    bs = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)
    test_loader = DataLoader(test_dataset, batch_size=bs)

    in_channels = train_dataset[0].x.size(1) if train_dataset[0].x is not None else 1

    # Determine output channels from data
    sample = train_dataset[0]
    if config["task"] == "graph_regression":
        out_channels = sample.y.size(-1) if sample.y.dim() > 0 else 1
    elif config["task"] == "graph_classification":
        out_channels = config["num_classes"]
    elif config["task"] == "node_classification":
        out_channels = config["num_classes"]
    else:
        out_channels = config["num_classes"]

    return {
        "data": None,
        "dataset": train_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "config": {
            **config,
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
    }
