# Code base for "Spectral Graph Pruning Against Over-squashing and Over-smoothing", which is now accepted at NeurIPS 2024!

The basic requirements for reproducing results are listed below. Alternatively, we also provide the environment.yml for the exact environment requirements.

```Python
Pytorch-Geometric == 2.5.3
Pytorch ==2.2.1
DGL == 2.1.0+cu118
```

To run the code -

```bash
bash gen_spect.sh
```
The NodeClassification/rewiring folder contains

1. Proxyaddmax - add edges by maximizing proxy spectral gap.
2. Proxyaddmin - add edges by minmizing proxy spectral gap.
3. Proxydelmax - delete edges by maximizing proxy spectral gap.
4. Proxydelmin - delete edges by maximizing proxy spectral gap.
5. FoSR - add edges by maximizing first order approximation of spectral gap proposed in https://openreview.net/forum?id=3YjQfCLdrzz.
   
Datasets Cora,Citeseer,Pubmed are automatically downloaded from Pytorch-Geometric. Cornell,Texas,Wisconsin,Chameleon,Squirrel and Actor datasets we use the .npz files given here - https://github.com/yandex-research/heterophilous-graphs



