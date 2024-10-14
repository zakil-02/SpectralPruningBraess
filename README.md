# Code base for Spectral Graph Pruning Against Over-squashing and Over-smoothing which is now accepted at NeurIPS 2024!

The basic requirements for reproducing results are listed below. Alternatively, we also provide the environment.yml for the exact environment requirements.

```Python
Pytorch-Geometric == 2.5.3
Pytorch ==2.2.1
DGL == 2.1.0+cu118
```

To run the code -

```Python
python main.py --dataset 'Cora' --method 'proxydelmax' --max_iters 50 --out 'Planetoid.csv' --existing_graph None
```

Supports -

1. Rewiring using proxyaddmin,proxyaddmax,proxydelmin,proxydelmax. Add/Delete edges to either maximize or minimize the spectral gap.
2. Rewiring using FoSR. Adds edges to maximize spectral gap.




