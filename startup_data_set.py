#!/usr/bin/env python
# coding: utf-8

# # Startup Data class

# In[1]:


import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
import dgl
import scipy.sparse as sp
device = torch.device('cpu')


# In[2]:


class COMP4222Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='comp-4222')
    def process(self):
        self.df_startups = pd.read_csv('./data/startups_formatted.csv')
        self.df_investors = pd.read_csv('./data/investors_formatted.csv')
        self.df_investments = pd.read_csv('./data/funding_round_formatted.csv')
       
        # drop unlinked node
        self.df_startups = self.df_startups.drop(i for i in self.df_startups.id.values.tolist() if i not in self.df_investments.funded_object_id.values.tolist())
        self.df_startups = self.df_startups.reset_index()
        
        dictionary = dict(zip(np.unique(self.df_investments.funded_object_id.values),self.df_startups.index.values))
        self.df_investments['investor_object_id'] = self.df_investments['investor_object_id'] + len(self.df_startups)
        self.df_investments["funded_object_id"] = self.df_investments["funded_object_id"].replace(dictionary)
        
        self.df_investments = self.df_investments.groupby(['investor_object_id','funded_object_id']).sum()
        self.df_investments = self.df_investments.reset_index()
        self.investments_edge = len(self.df_investments)
        
        self.startup_node = len(self.df_investments)
        self.investor_node = len(self.df_investors)
        
        self.graph = dgl.graph((torch.tensor(self.df_investments.funded_object_id.values.tolist()), 
                                torch.tensor(self.df_investments.investor_object_id.values.tolist())))

        
        
        self.graph.ndata['feat'] = torch.concat((torch.tensor(self.df_startups.iloc[:, 3:].to_numpy()), 
                                                 torch.tensor(np.pad(self.df_investors.iloc[:, 2:].to_numpy(), 
                                                                     [(0,0),(0,120)], 
                                                                     mode='constant', constant_values=0))))
        # 0 for startup, 1 for investor
        self.graph.ndata['label'] = torch.concat((torch.zeros(len(self.df_startups)), 
                                                  torch.ones(len(self.df_investors))))

        edge_feature = [i for i in self.df_investments.columns if i not in ["funding_round_id", "funded_object_id", "investor_object_id"]]
        self.graph.edata['feat'] = torch.tensor(self.df_investments[edge_feature].to_numpy())
  

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


# In[66]:


class COMP4222Dataset_hetero(DGLDataset):
    def __init__(self):
        super().__init__(name='comp-4222')
    def process(self):
        self.df_startups = pd.read_csv('./data/startups_formatted.csv')
        self.df_investors = pd.read_csv('./data/investors_formatted.csv')
        self.df_investments = pd.read_csv('./data/funding_round_formatted.csv')
       
        # drop unlinked node
        self.df_startups = self.df_startups.drop(i for i in self.df_startups.id.values.tolist() if i not in self.df_investments.funded_object_id.values.tolist())
        self.df_startups = self.df_startups.reset_index()
        
        # replace fundedobject id in investment to 
        dictionary = dict(zip(np.unique(self.df_investments.funded_object_id.values),self.df_startups.index.values))
        self.df_investments["funded_object_id"] = self.df_investments["funded_object_id"].replace(dictionary)
        
        self.df_investments = self.df_investments.groupby(['investor_object_id','funded_object_id']).sum()
        self.df_investments = self.df_investments.reset_index()
        self.investments_edge = len(self.df_investments)
        
        self.startup_node = len(self.df_investments)
        self.investor_node = len(self.df_investors)
        self.graph = dgl.heterograph(
            {
                ("investor", "i_s", "startup"): (torch.tensor(self.df_investments.investor_object_id.values.tolist())
                                                 ,torch.tensor(self.df_investments.funded_object_id.values.tolist())),
            }
        )

        self.graph.nodes["investor"].data['feat'] = torch.tensor(np.pad(self.df_investors.iloc[:, 2:].to_numpy(), [(0,0),(0,120)], mode='constant', constant_values=0))
        self.graph.nodes["startup"].data['feat'] = torch.tensor(self.df_startups.iloc[:, 3:].to_numpy())
        edge_feature = [i for i in self.df_investments.columns if i not in ["funding_round_id", "funded_object_id", "investor_object_id"]]
        self.graph.edata['feat'] = torch.tensor(self.df_investments[edge_feature].to_numpy())
  

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

