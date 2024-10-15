# Code base for "Spectral Graph Pruning Against Over-squashing and Over-smoothing", which is now accepted at NeurIPS 2024!

>Message Passing Graph Neural Networks are known to suffer from two problems that are sometimes believed to be diametrically opposed: over-squashing and over-smoothing. The former results from topological bottlenecks that hamper the information flow from distant nodes and are mitigated by spectral gap maximization, primarily, by means of edge additions. However, such additions often promote over-smoothing that renders nodes of different classes less distinguishable. Inspired by the Braess phenomenon, we argue that deleting edges can address over-squashing and over-smoothing simultaneously. This insight explains how edge deletions can improve generalization, thus connecting spectral gap optimization to a seemingly disconnected objective of reducing computational resources by pruning graphs for lottery tickets. To this end, we propose a computationally effective spectral gap optimization framework to add or delete edges and demonstrate its effectiveness on the long range graph benchmark and on larger heterophilous datasets.

![](https://github.com/AdarshMJ/SpectralPruningBraess/blob/main/BraessRing.jpg)



## Requirements
The basic library requirements for reproducing results are listed below. 

```Python
Pytorch-Geometric == 2.5.3
Pytorch ==2.2.1
DGL == 2.1.0+cu118
```
Alternatively, we also provide the environment.yml for the exact environment requirements.

To run the code -

## Structure of the code
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

## Citation
```bibtex
@inproceedings{jamadandi2024spectral,
title={Spectral Graph Pruning Against Over-Squashing and Over-Smoothing},
author={Adarsh Jamadandi and Celia Rubio-Madrigal and Rebekka Burkholz},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```







