#!/usr/bin/env python
# coding: utf-8

# # Startup Data class

# In[80]:


import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
import dgl
device = torch.device('cpu')


# In[92]:


class COMP4222Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='comp-4222')

    def process(self):
        self.df_startups = pd.read_csv('./data/startups_formatted.csv')
        self.df_investors = pd.read_csv('./data/investors_formatted.csv')
        self.df_investments = pd.read_csv('./data/funding_round_formatted.csv')
        self.df_investors['object_id'] = self.df_investors['object_id'] + len(self.df_startups)
        self.df_investments['investor_object_id'] = self.df_investments['investor_object_id'] + len(self.df_startups)
        self.startup_node = len(self.df_startups)
        self.investor_node = len(self.df_investors)
        self.investments_edge = len(self.df_investments)

        self.graph = dgl.graph((torch.tensor(self.df_investments.funded_object_id.values.tolist()), torch.tensor(self.df_investments.investor_object_id.values.tolist())))
        self.graph.ndata['feat'] = torch.concat((torch.tensor(self.df_startups.iloc[:, 2:].to_numpy()), torch.tensor(np.pad(self.df_investors.iloc[:, 2:].to_numpy(), [(0,0),(0,120)], mode='constant', constant_values=0))))
        # 0 for startup, 1 for investor
        self.graph.ndata['label'] = torch.concat((torch.zeros(len(self.df_startups)), torch.ones(len(self.df_investors))))

        edge_feature = [i for i in self.df_investments.columns if i not in ["funding_round_id", "funded_object_id", "investor_object_id"]]
        self.graph.edata['feat'] = torch.tensor(self.df_investments[edge_feature].to_numpy())
  

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = COMP4222Dataset()
graph = dataset[0]

print(graph)


# In[ ]:





# In[ ]:




