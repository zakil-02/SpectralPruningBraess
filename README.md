# Code base for "Spectral Graph Pruning Against Over-squashing and Over-smoothing", which is now accepted at NeurIPS 2024!

We propose a computationally efficient proxy spectral gap based graph rewiring strategy inspired by the Braess Paradox to show we can alleviate both over-squashing and over-smoothing in graph neural networks!

![](https://github.com/AdarshMJ/SpectralPruningBraess/blob/main/BraessRing.jpg)




The basic library requirements for reproducing results are listed below. 

```Python
Pytorch-Geometric == 2.5.3
Pytorch ==2.2.1
DGL == 2.1.0+cu118
```
Alternatively, we also provide the environment.yml for the exact environment requirements.

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
   
Datasets - Cora,Citeseer,Pubmed are automatically downloaded from Pytorch-Geometric. 

For Cornell, Texas, Wisconsin, Chameleon, Squirrel and Actor we use the .npz files given here - https://github.com/yandex-research/heterophilous-graphs

For generating results for large heterophilic graphs -
1. Download the code base - https://github.com/yandex-research/heterophilous-graphs
2. Install DGL
3. Change datasets.py to include the rewiring schemes. We have provided with the datasets.py file that needs to be replaced and can be used to perform graph rewiring on the large heterophilic graphs.
4. Change the LR = 3e-4 or 3e-3 to get better results.


If you found this paper interesting and the code base helpful please consider citing :


```bibtex
@inproceedings{jamadandi2024spectral,
title={Spectral Graph Pruning Against Over-Squashing and Over-Smoothing},
author={Adarsh Jamadandi and Celia Rubio-Madrigal and Rebekka Burkholz},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```







