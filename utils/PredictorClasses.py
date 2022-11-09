#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            # save in graph node data
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(2*h_feats , h_feats).to(torch.float32)
        self.W2 = nn.Linear(h_feats, 1).to(torch.float32)

    # concat the source and destination node, use mlp to predict the score
    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1).to(torch.float32)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h.to(torch.float32)
            g.apply_edges(self.apply_edges)
            return g.edata['score']

