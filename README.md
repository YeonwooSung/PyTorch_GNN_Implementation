# PyTorch GNN Implementation

My implementation of GNNs in PyTorch.

## Table of Contents

- [Setup](#setup)
- [Models](#models)

## Setup

```bash
pip install -r requirements.txt
```

## Models

- [Graph Convolutional Network](./src/gcn/)
- [GraphSAGE](./src/graphsage/)
- [GAT](./src/gat/)
- [GAT v2](./src/gat-v2/)

## Experiments

### Overfitting issue with GAT v2

While training the GAT and GAT-v2 with Cora, I found that the GAT-v2 easily overfitted. [S. Brody et. al. [3]](https://arxiv.org/abs/2105.14491) stated that "Intuitively, we believe that the more complex the interactions between nodes are – the more benefit a GNN can take from theoretically stronger graph attention mechanisms such as GATv2. The main question is whether the problem has a global ranking of “influential” nodes (GAT is sufficient), or do different nodes have different rankings of neighbors (use GATv2)". From here, I am assuming that the GAT-v2 model is too complex to be used on Cora dataset.

## ToDo

[v] Add training codes for GraphSAGE

## References

[1] Thomas N. Kipf, Max Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

[2] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[3] Shaked Brody, Uri Alon, Eran Yahav. [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)

[4] William L. Hamilton, Rex Ying, Jure Leskovec. [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
