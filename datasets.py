"""
Dataset loading utilities for QGNN experiments.

Supports:
- Standard benchmarks: Cora, Citeseer, PPI, Circuits
- Long Range Graph Benchmark (LRGB): PascalVOC-SP, COCO-SP, Peptides-func,
  Peptides-struct, PCQM-Contact
"""

import os

import torch
from torch_geometric.datasets import Planetoid, PPI, LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures


DATASET_CONFIGS = {
    # Standard benchmarks
    "cora": {
        "task": "node_classification",
        "metric": "accuracy",
        "num_classes": 7,
        "max_epochs": 200,
        "batch_size": None,  # Full-batch
        "fixed_n": 2708,
    },
    "citeseer": {
        "task": "node_classification",
        "metric": "accuracy",
        "num_classes": 6,
        "max_epochs": 200,
        "batch_size": None,
        "fixed_n": 3327,
    },
    "ppi": {
        "task": "multilabel",
        "metric": "micro_f1",
        "num_classes": 121,
        "max_epochs": 200,
        "batch_size": 2,
        "fixed_n": None,
    },
    "circuits": {
        "task": "graph_regression",
        "metric": "mae",
        "num_classes": 1,
        "max_epochs": 200,
        "batch_size": 32,
        "fixed_n": None,
    },
    # LRGB benchmarks
    "pascalvoc-sp": {
        "task": "node_classification",
        "metric": "f1",
        "num_classes": 21,
        "max_epochs": 500,
        "batch_size": 32,
        "fixed_n": None,
        "node_sample_size": 256,
    },
    "coco-sp": {
        "task": "node_classification",
        "metric": "f1",
        "num_classes": 81,
        "max_epochs": 500,
        "batch_size": 32,
        "fixed_n": None,
        "node_sample_size": 256,
    },
    "peptides-func": {
        "task": "multilabel",  # Fix #3: multi-label, not single-label classification
        "metric": "ap",
        "num_classes": 10,
        "max_epochs": 500,
        "batch_size": 32,
        "fixed_n": None,
    },
    "peptides-struct": {
        "task": "graph_regression",
        "metric": "mae",
        "num_classes": 11,
        "max_epochs": 500,
        "batch_size": 32,
        "fixed_n": None,
    },
    "pcqm-contact": {
        "task": "link_prediction",
        "metric": "hits_at_10",
        "num_classes": 1,
        "max_epochs": 500,
        "batch_size": 32,
        "fixed_n": None,
    },
}


def load_dataset(name: str, root: str = "./data"):
    """Load a dataset by name.

    Args:
        name: Dataset name (case-insensitive).
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
    elif name == "circuits":
        return _load_circuits(root, config)
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


def _load_circuits(root: str, config: dict) -> dict:
    """Load the Electronic Circuits (CktGNN) operational amplifier dataset.

    The raw data is distributed by the CktGNN authors (Dong et al., 2023) at
    https://github.com/zehao-dong/CktGNN. This loader expects the data to have
    been downloaded and converted to a list of torch_geometric.data.Data
    objects saved as train.pt / val.pt / test.pt under {root}/Circuits/.
    """
    import os.path as osp

    from torch_geometric.data import InMemoryDataset, Data

    circuits_root = osp.join(root, "Circuits")
    train_path = osp.join(circuits_root, "train.pt")
    val_path = osp.join(circuits_root, "val.pt")
    test_path = osp.join(circuits_root, "test.pt")

    if not (osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path)):
        raise FileNotFoundError(
            f"Circuits dataset files not found in {circuits_root}. "
            f"Download the CktGNN operational amplifier data from "
            f"https://github.com/zehao-dong/CktGNN and preprocess into "
            f"train.pt / val.pt / test.pt containing lists of "
            f"torch_geometric.data.Data objects."
        )

    class _CircuitsDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__(".")
            self.data, self.slices = self.collate(data_list)

    train_list = torch.load(train_path, weights_only=False)
    val_list = torch.load(val_path, weights_only=False)
    test_list = torch.load(test_path, weights_only=False)

    train_dataset = _CircuitsDataset(train_list)
    val_dataset = _CircuitsDataset(val_list)
    test_dataset = _CircuitsDataset(test_list)

    bs = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)
    test_loader = DataLoader(test_dataset, batch_size=bs)

    sample = train_dataset[0]
    in_channels = sample.x.size(1) if sample.x is not None else 1
    out_channels = sample.y.size(-1) if sample.y.dim() > 0 else 1

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


def _load_lrgb(name: str, root: str, config: dict) -> dict:
    """Load an LRGB dataset with official train/val/test splits."""
    lrgb_root = os.path.join(root, "LRGB")

    train_dataset = LRGBDataset(root=lrgb_root, name=name, split="train")
    val_dataset = LRGBDataset(root=lrgb_root, name=name, split="val")
    test_dataset = LRGBDataset(root=lrgb_root, name=name, split="test")

    # PCQM-Contact is a link prediction task and requires edge-level labels.
    # PyG's LRGBDataset.process_pcqm_contact() attaches these attributes to
    # every Data object; verify at load time.
    if name == "pcqm-contact":
        sample = train_dataset[0]
        if not (hasattr(sample, "edge_label_index") and hasattr(sample, "edge_label")):
            raise RuntimeError(
                "PCQM-Contact samples do not expose 'edge_label_index' and "
                "'edge_label'. Please upgrade torch-geometric to a recent "
                "version (tested with 2.7.0)."
            )

    bs = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)
    test_loader = DataLoader(test_dataset, batch_size=bs)

    in_channels = train_dataset[0].x.size(1) if train_dataset[0].x is not None else 1

    # Determine output channels from data
    sample = train_dataset[0]
    if config["task"] == "graph_regression":
        out_channels = sample.y.size(-1) if sample.y.dim() > 0 else 1
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
