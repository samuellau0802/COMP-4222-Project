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
    adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense()
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










def generate_train_test_valid_hetero_graph(graph, test_ratio=0.1,valid_ratio = 0.1, etype=("investor","raise","startup")):
    utype, efeat, vtype = etype
    u, v = graph.edges()
    eids = np.arange(graph.number_of_edges())
    eids = np.random.permutation(eids)
    src_feats = graph.nodes[utype].data['feat']
    dst_feats = graph.nodes[vtype].data['feat']
    edge_feats = graph.edges[efeat].data['feat']

    
    test_size = int(test_ratio*len(eids))
    valid_size = int(valid_ratio*len(eids))
    train_size = len(eids) - test_size - valid_size
    
    
    
    valid_pos_u, valid_pos_v = u[eids[:valid_size]], v[eids[:valid_size]]
    test_pos_u, test_pos_v = u[eids[valid_size:valid_size+test_size]], v[eids[valid_size:valid_size+test_size]]
    train_pos_u, train_pos_v = u[eids[valid_size+test_size:]], v[eids[valid_size+test_size:]]
    
    
    train_edge_features = edge_feats[eids[valid_size+test_size:]]
    test_edge_features = edge_feats[eids[valid_size:valid_size+test_size]]
    valid_edge_features = edge_feats[eids[:valid_size]]
    
    train_g = dgl.heterograph({etype: (train_pos_u, train_pos_v)})
    test_g = dgl.heterograph({etype: (test_pos_u, test_pos_v)})
    valid_g = dgl.heterograph({etype: (valid_pos_u, valid_pos_v)})
    
    # add node feature in train graph
    ti = train_g.nodes("investor").tolist()
    ts = train_g.nodes("startup").tolist()
    add_node_ti = []
    add_node_ts = []
    for i in ti:
        if i in graph.nodes(utype).tolist():
            add_node_ti.append(src_feats[i])
    for j in ts:
        if j in graph.nodes(vtype).tolist():
            add_node_ts.append(dst_feats[j])
            

    train_g.nodes[utype].data['feat']=torch.stack(add_node_ti)        
    train_g.nodes[vtype].data['feat']=torch.stack(add_node_ts)  
    
    # add node feature in test graph
    ti = test_g.nodes("investor").tolist()
    ts = test_g.nodes("startup").tolist()
    add_node_ti = []
    add_node_ts = []
    for i in ti:
        if i in graph.nodes(utype).tolist():
            add_node_ti.append(src_feats[i])
    for j in ts:
        if j in graph.nodes(vtype).tolist():
            add_node_ts.append(dst_feats[j])
    test_g.nodes[utype].data['feat']=torch.stack(add_node_ti)        
    test_g.nodes[vtype].data['feat']=torch.stack(add_node_ts)  
    
    # add node feature in valid graph
    ti = valid_g.nodes("investor").tolist()
    ts = valid_g.nodes("startup").tolist()
    add_node_ti = []
    add_node_ts = []
    for i in ti:
        if i in graph.nodes(utype).tolist():
            add_node_ti.append(src_feats[i])
    for j in ts:
        if j in graph.nodes(vtype).tolist():
            add_node_ts.append(dst_feats[j])

    valid_g.nodes[utype].data['feat']=torch.stack(add_node_ti)        
    valid_g.nodes[vtype].data['feat']=torch.stack(add_node_ts)
    
    
    ####### ADD EDGE FEATURE IN TRAIN TEST VALID GRAPH##############
    # create edge feature dict for add edge feature to created graph
    key = list(zip(u.tolist(),v.tolist()))
    edge_features_dict = dict(zip(key,edge_feats))
    
    # add tain set test data
    train_u,train_v = train_g.edges()
    train_u,train_v = train_u.tolist(),train_v.tolist()
    add_edge=[]
    for key in list(zip(train_u,train_v)):
        if key in edge_features_dict.keys():
            add_edge.append(edge_features_dict.get(key))
    train_g.edges[efeat].data['feat'] = torch.stack(add_edge)

    # add test set test data
    test_u,test_v = test_g.edges()
    test_u,test_v = test_u.tolist(),test_v.tolist()
    add_edge=[]
    for key in list(zip(test_u,test_v)):
        if key in edge_features_dict.keys():
            add_edge.append(edge_features_dict.get(key))
    test_g.edges[efeat].data['feat'] = torch.stack(add_edge)
    

    # add valid set edge data
    valid_u,valid_v = valid_g.edges()
    valid_u,valid_v = valid_u.tolist(),valid_v.tolist()
    add_edge=[]
    for key in list(zip(valid_u,valid_v)):
        if key in edge_features_dict.keys():
            add_edge.append(edge_features_dict.get(key))
    valid_g.edges[efeat].data['feat'] = torch.stack(add_edge)
    
    return train_g, test_g, valid_g

def construct_negative_hetero_graph(graph, k, etype):
    # k = 5
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)})