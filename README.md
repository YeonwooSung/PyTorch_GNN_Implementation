# PyTorch GNN Implementation

My implementation of GNNs in PyTorch.

## Table of Contents

- [Setup](#setup)
- [Experiments](#experiments)
- [Models](#models)
    * [GCN](#gcn)
    * [GraphSAGE](#graphsage)
    * [GAT](#gat)
    * [GAT-v2](#gat-v2)
- [References](#references)

## Setup

```bash
pip install -r requirements.txt
```

## Experiments

### Overfitting issue with GAT v2

While training the GAT and GAT-v2 with Cora, I found that the GAT-v2 easily overfitted. [S. Brody et. al. [3]](https://arxiv.org/abs/2105.14491) stated that "Intuitively, we believe that the more complex the interactions between nodes are – the more benefit a GNN can take from theoretically stronger graph attention mechanisms such as GATv2. The main question is whether the problem has a global ranking of “influential” nodes (GAT is sufficient), or do different nodes have different rankings of neighbors (use GATv2)". From here, I am assuming that the GAT-v2 model is too complex to be used on Cora dataset.

### GraphSAGE works better than GAT

With Cora dataset for node classification task, GAT got 81.3% accuracy, where GraphSAGE got 82.1% accuracy. Furthermore, the GraphSAGE model achieved 83.5% accuracy with Citeseer dataset, while GAT got 81.9% accuracy. From here, I am assuming that GraphSAGE is better than GAT for node classification task.

Also, GraphSAGE achieved 88.6% for node classification task with Pubmed dataset. Did not tested GAT with Pubmed dataset, but I am assuming that GraphSAGE is better than GAT for node classification task.

## Models

### GCN

_Semi-Supervised Classification with Graph Convolutional Networks_

Authors: Thomas N. Kipf, Max Welling

[src](./src/gcn/) [paper](https://arxiv.org/abs/1609.02907)

#### Abstract

We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.

### GraphSAGE

_Inductive Representation Learning on Large Graphs_

Authors: William L. Hamilton, Rex Ying, Jure Leskovec

[src](./src/graphsage/) [paper](https://arxiv.org/abs/1706.02216)

#### Abstract

Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

### GAT

_Graph Attention Networks_

Authors: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio

[src](./src/gat/) [paper](https://arxiv.org/abs/1710.10903)

#### Abstract

We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

### GAT-v2

_How Attentive are Graph Attention Networks?_

Authors: Shaked Brody, Uri Alon, Eran Yahav

[GAT v2](./src/gat-v2/) [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)

#### Abstract

Graph Attention Networks (GATs) are one of the most popular GNN architectures and are considered as the state-of-the-art architecture for representation learning with graphs. In GAT, every node attends to its neighbors given its own representation as the query. However, in this paper we show that GAT computes a very limited kind of attention: the ranking of the attention scores is unconditioned on the query node. We formally define this restricted kind of attention as static attention and distinguish it from a strictly more expressive dynamic attention. Because GATs use a static attention mechanism, there are simple graph problems that GAT cannot express: in a controlled problem, we show that static attention hinders GAT from even fitting the training data. To remove this limitation, we introduce a simple fix by modifying the order of operations and propose GATv2: a dynamic graph attention variant that is strictly more expressive than GAT. We perform an extensive evaluation and show that GATv2 outperforms GAT across 11 OGB and other benchmarks while we match their parametric costs. Our code is available at this https URL . GATv2 is available as part of the PyTorch Geometric library, the Deep Graph Library, and the TensorFlow GNN library.

## ToDo

## References

[1] Thomas N. Kipf, Max Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

[2] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[3] Shaked Brody, Uri Alon, Eran Yahav. [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)

[4] William L. Hamilton, Rex Ying, Jure Leskovec. [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
