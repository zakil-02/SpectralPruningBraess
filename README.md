# SpectralPruningBraess

Requirements

```
Pytorch-Geometric
Pytorch
Networkx
```

To run the code -

```Python
python main.py --dataset 'Cora' --method 'proxydelmin' --max_iters 50 --out 'Planetoid.csv' --existing_graph None
```

Supports -

1. Rewiring using proxyaddmin,proxyaddmax,proxydelmin,proxydelmax. Add/Delete edges to either maximize or minimize the spectral gap.
2. Rewiring using FoSR. Adds edges to maximize spectral gap.
3. Rewiring using SDRF. Rewires the graph based on discrete Ricci curvature.



