# QGNN: Quantum-Inspired Graph Neural Network with Quantum Entanglement Loss

Official implementation of **"Quantum-Enhanced Learning: Leveraging Von Neumann Entropy for Enhanced Graph Neural Network Performance"** (Neural Networks, 2026).

## Overview

QGNN addresses the **over-squashing** problem in Graph Neural Networks by introducing a novel **Quantum Entanglement Loss (QEL)** function. QEL minimizes the von Neumann entropy of the node embedding correlation matrix, concentrating eigenvalues in dominant eigenmodes that preserve long-range structural information.

### Key Features

- **Quantum Entanglement Loss**: Regularization based on von Neumann entropy minimization of the density matrix constructed from node embeddings
- **Global Correlation Module**: Each QGNN layer combines local GCN message-passing with a global quantum-inspired correlation term
- **Efficient Computation**: Top-k eigenvalue approximation via power iteration reduces complexity from O(n^3) to O(kn^2)
- **5-6x faster** than Graph Transformer approaches with comparable or better performance

## Architecture

Each QGNN layer implements:

```
H^(l+1) = sigma( A_norm H^(l) W^(l) + beta * rho^(l) H^(l) W_V^(l) )
```

where `rho^(l) = (1/n) H^(l) (H^(l))^T` is the density matrix at layer l, followed by L2 row normalization.

The total loss combines the task loss with QEL:

```
L_total = L_task + alpha * S(rho)
```

where `S(rho) = -sum_i lambda_i log(lambda_i)` is the von Neumann entropy.

## Installation

```bash
# Clone the repository
git clone https://github.com/muhammadawais95/QGNN.git
cd QGNN

# Install dependencies
pip install -r requirements.txt
```

**Note**: PyTorch Geometric installation depends on your CUDA version. See [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

## Usage

### Quick Start

```bash
# Node classification on Cora
python train.py --dataset cora --alpha 0.1 --beta 0.1

# Node classification on Citeseer
python train.py --dataset citeseer --alpha 0.1 --beta 0.1

# Graph regression on Peptides-struct (LRGB)
python train.py --dataset peptides-struct --alpha 0.15 --epochs 500

# Graph classification on Peptides-func (LRGB)
python train.py --dataset peptides-func --alpha 0.1 --epochs 500

# Node classification on PascalVOC-SP (LRGB)
python train.py --dataset pascalvoc-sp --alpha 0.1 --epochs 500 --k 32
```

### Reproduce All Experiments

```bash
bash experiments/run_all.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.1 | QEL weight (0.08-0.12 optimal for most datasets; 0.15-0.2 for large-diameter graphs) |
| `--beta` | 0.1 | Global correlation strength in each layer |
| `--k` | 16 | Number of top eigenvalues for entropy approximation (16 for small, 32 for large graphs) |
| `--hidden_channels` | 16 | Hidden layer dimension |
| `--num_layers` | 2 | Number of QGNN layers |
| `--dropout` | 0.2 | Dropout rate |
| `--lr` | 0.001 | Learning rate |
| `--weight_decay` | 5e-4 | L2 regularization |
| `--patience` | 30 | Early stopping patience |
| `--runs` | 10 | Number of evaluation runs |

## Project Structure

```
QGNN/
├── model.py          # QGNN model (QGNNLayer, QGNN, task-specific variants)
├── train.py          # Training loop with evaluation
├── datasets.py       # Data loading for all benchmarks
├── utils.py          # QEL computation (entropy, power iteration, density matrix)
├── requirements.txt  # Python dependencies
├── experiments/
│   └── run_all.sh    # Script to reproduce all paper experiments
└── README.md
```

## Supported Datasets

### Standard Benchmarks
| Dataset | Task | Metric | Nodes | Avg Diameter |
|---------|------|--------|-------|-------------|
| Cora | Node Classification | Accuracy | 2,708 | 6.31 |
| Citeseer | Node Classification | Accuracy | 3,327 | 7.57 |
| PPI | Multi-label Classification | micro-F1 | 56,944 | 4.82 |

### Long Range Graph Benchmark (LRGB)
| Dataset | Task | Metric | Graphs | Avg Diameter |
|---------|------|--------|--------|-------------|
| PascalVOC-SP | Node Classification | macro-F1 | 11,355 | 27.62 |
| COCO-SP | Node Classification | macro-F1 | 123,286 | 27.39 |
| Peptides-func | Graph Classification | AP | 15,535 | 56.99 |
| Peptides-struct | Graph Regression | MAE | 15,535 | 56.99 |
| PCQM-Contact | Link Prediction | Hits@10 | 529,434 | 4.63* |

\*Average shortest path

## Citation

If you use this code, please cite:

```bibtex
@article{awais2026quantum,
  title={Quantum-Enhanced Learning: Leveraging Von Neumann Entropy for Enhanced Graph Neural Network Performance},
  author={Awais, Muhammad and Postolache, Octavian Adrian and Oliveira, Sancho Moura},
  journal={Neural Networks},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License.

