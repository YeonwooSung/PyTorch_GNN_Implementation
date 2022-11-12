import scipy.sparse as sp


def get_adjacent_matrix_from_node_and_edge_matrices(node_matrix, edge_matrix):
    """Get adjacent matrix from node and edge matrices.

    Parameters
    ----------
    node_matrix : Node matrix.
    edge_matrix : Edge matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacent matrix.
    """
    node_spr_matrix = sp.csr_matrix(node_matrix)
    edge_spr_matrix = sp.csr_matrix(edge_matrix)
    return node_spr_matrix.dot(edge_spr_matrix).dot(node_spr_matrix.T)
