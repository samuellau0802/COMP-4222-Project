#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.cpu(), neg_score.cpu()]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

