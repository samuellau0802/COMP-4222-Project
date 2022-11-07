#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd


# In[3]:


class COMP4222Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='comp-4222')

    def process(self):
        df_startups = pd.read_csv('./data/startups_formatted.csv')
        df_investors = pd.read_csv('./data/investors_formatted.csv')
        df_investments = pd.read_csv('./data/funding_round_formatted.csv')

        data_dict = {
            ("investor", "raise", "startup"): (torch.tensor(df_investments.investor_object_id.values.tolist()), torch.tensor(df_investments.funded_object_id.values.tolist()))
            }     
        self.graph = dgl.heterograph(data_dict)
        
        edge_feature = [i for i in df_investments.columns if i not in ["funding_round_id", "funded_object_id", "investor_object_id"]]

        self.graph.nodes['investor'].data['feat'] = torch.tensor(df_investors.iloc[:, 2:].to_numpy())
        self.graph.nodes['startup'].data['feat'] = torch.tensor(df_startups.iloc[:, 2:].to_numpy())

        self.graph.edges['raise'].data['feat'] = torch.tensor(df_investments[edge_feature].to_numpy())

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = COMP4222Dataset()
graph = dataset[0]

print(graph)


# In[4]:


print(f"Startup Node Size:  {graph.nodes['startup'].data['feat'].shape}")
print(f"Investor Node Size:  {graph.nodes['investor'].data['feat'].shape}")
print(f"Edge Size:  {graph.edges['raise'].data['feat'].shape}")


# In[5]:


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp


# In[ ]:




