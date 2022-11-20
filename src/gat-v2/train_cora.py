import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
import pickle

from gat_v2 import GATV2


CORA_PATH = './cora/'
CORA_TRAIN_RANGE = [0, 140]  # we're using the first 140 nodes as the training nodes
CORA_VAL_RANGE = [140, 140+500]
CORA_TEST_RANGE = [1708, 1708+1000]



def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    # shape = (N, FIN) -> (N, 1), where N number of nodes and FIN number of input features
    node_features_sum = np.array(node_features_sparse.sum(-1))  # sum features for every node feature vector

    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    # shape = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 1 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.

    # Create a diagonal matrix whose values on the diagonal come from node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # We return the normalized features.
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index


def load_graph_data(device):
    # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
    node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
    # shape = (N, 1)
    node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
    # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
    adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

    # Normalize the features (helps with training)
    node_features_csr = normalize_features_sparse(node_features_csr)
    num_of_nodes = len(node_labels_npy)

    # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
    # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
    topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)

    # generate adjacency matrix from edge index
    # shape = (N, N, 1), where N is the number of nodes
    adjacency_matrix = sp.coo_matrix((np.ones(topology.shape[1]), (topology[0], topology[1])), shape=(num_of_nodes, num_of_nodes))
    # adjacency_matrix to numpy array
    adjacency_matrix = adjacency_matrix.toarray()
    adjacency_matrix = adjacency_matrix[:, :, np.newaxis]
    
    # adjacency_matrix to tensor
    adjacency_matrix = torch.from_numpy(adjacency_matrix).to(device)


    # Note: topology is just a fancy way of naming the graph structure data 
    # (aside from edge index it could be in the form of an adjacency matrix)

    # Convert to dense PyTorch tensors

    # Needs to be long int type because later functions like PyTorch's index_select expect it
    topology = torch.tensor(topology, dtype=torch.long, device=device)
    node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)  # Cross entropy expects a long int
    node_features = torch.tensor(node_features_csr.todense(), device=device)

    # Indices that help us extract nodes that belong to the train/val and test splits
    train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
    val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
    test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

    return node_features, node_labels, topology, train_indices, val_indices, test_indices, adjacency_matrix


def train(
    epochs: int,
    patience_period: int,
    in_features: int,
    n_hidden: int,
    n_classes: int,
    n_heads: int,
    dropout: float,
    num_of_layers: int,
    optimizer_lr: float = 5e-3,
    optimizer_weight_decay: float = 5e-4,
    print_every: int = 100,
    patience_count_interval: int = 100,
):
    model = GATV2(
        in_features=in_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        n_heads=n_heads,
        dropout=dropout,
        num_of_layers=num_of_layers,
        share_weights=True
    )
    optimizer = Adam(model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    node_features, node_labels, topology, train_indices, val_indices, test_indices, adjacency_list_dict = load_graph_data(device)

    patience_counter = 0
    best_score = 0.
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(node_features, adjacency_list_dict)
        loss = F.cross_entropy(logits[train_indices], node_labels[train_indices])
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

        if epoch % patience_count_interval == 0:
            class_predictions_t = torch.argmax(logits[train_indices], dim=-1)
            class_predictions_v = torch.argmax(logits[val_indices], dim=-1)
            train_accuracy = torch.sum(torch.eq(class_predictions_t, node_labels[train_indices]).long()).item() / len(node_labels[train_indices])
            valid_accuracy = torch.sum(torch.eq(class_predictions_v, node_labels[val_indices]).long()).item() / len(node_labels[val_indices])

            print(f'Train accuracy: {train_accuracy}, Val accuracy: {valid_accuracy}')

            if valid_accuracy > best_score:
                best_score = valid_accuracy
                patience_counter = 0
            else:
                patience_counter += patience_count_interval
        
        if patience_counter >= patience_period:
            print(f'Early stopping at epoch {epoch}')
            break


if __name__ == '__main__':
    train(
        epochs=10000,
        patience_period=1000,
        in_features=1433,
        n_hidden=8,
        n_classes=7,
        n_heads=8,
        dropout=0.6,
        num_of_layers=2,
        optimizer_lr=5e-3,
        optimizer_weight_decay=5e-4,
    )
