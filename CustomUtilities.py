import numpy as np
import scipy.sparse as sp
import torch
import dgl

def generate_pos_graph(graph, val_ratio=0.1, test_ratio=0.1):
    '''Input the graph, return a tuple of 4 graphs in the form of (train_g, train_pos_g, val_pos_g, test_pos_g). It returns the positive graphs
    @param graph: the dgl graph
    @param val_ratio: the validation ratio. Default 80% train, 10% val, 10% test
    @param test_ratio: the test ratio. Default 80% train, 10% val, 10% test
    '''

    u, v = graph.edges()
    # give id for all edges then permutation
    eids = np.arange(graph.number_of_edges())
    eids = np.random.permutation(eids)

    test_size = int(len(eids) * test_ratio)
    val_size = int(len(eids) * val_ratio)

    val_u, val_v = u[eids[:val_size]], v[eids[:val_size]]
    test_u, test_v = u[eids[val_size:val_size+test_size]], v[eids[val_size:val_size+test_size]]
    train_u, train_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]

    train_g = dgl.remove_edges(graph, eids[:test_size+val_size])
    train_g = dgl.add_self_loop(train_g)
    train_pos_g = dgl.graph((train_u, train_v), num_nodes=graph.number_of_nodes())
    val_pos_g = dgl.graph((val_u, val_v), num_nodes=graph.number_of_nodes())
    test_pos_g = dgl.graph((test_u, test_v), num_nodes=graph.number_of_nodes())

    return train_g, train_pos_g, val_pos_g, test_pos_g

def generate_neg_graph(graph, val_ratio=0.1, test_ratio=0.1):
    '''Input the graph, return a tuple of 3 graphs in the form of (train_neg_g, val_neg_g, test_neg_g). It returns the negative graphs
    @param graph: the dgl graph
    @param val_ratio: the validation ratio. Default 80% train, 10% val, 10% test
    @param test_ratio: the test ratio. Default 80% train, 10% val, 10% test
    '''
    u, v = graph.edges()

    # Find all negative edges and split them for training and testing
    #use sparse matrix to save memory
    # ,shape = (torch.max(v)+1,torch.max(v)+1)
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(torch.max(u)+1,torch.max(v)+1)
    neg_u, neg_v = np.where(adj_neg != 0) # negative edge, we don't have edge
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())
    test_size = int(len(neg_eids) * test_ratio)
    val_size = int(len(neg_eids) * val_ratio)

    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    test_neg_u, test_neg_v = neg_u[neg_eids[val_size:test_size+val_size]], neg_v[neg_eids[val_size:test_size+val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]], neg_v[neg_eids[test_size+val_size:]]

    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_neg_g, val_neg_g, test_neg_g