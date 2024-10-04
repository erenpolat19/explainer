import torch
import argparse
import sys
import numpy as np
import models
import torch
import torch.nn.functional as F
import data_preprocessing
from torch.optim import Adam
from models import *
#from pretrain_clf import * 
import gcn
from data_preprocessing import *

from sklearn.metrics import roc_auc_score
import os 

def create_edge_embed(node_embeddings, edge_index):
    h_i = node_embeddings[edge_index[0]]  
    h_j = node_embeddings[edge_index[1]]  

    return torch.cat([h_i, h_j], dim=-1)

# gumbel-softmax reparam trick 
def sample_graph(sampling_weights, temperature=1.0, bias=0.0, device='cpu', training=True):
    if training:
        bias = bias + 0.0001  #apparently if bias is 0 there can be problems
        eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(),device=device) + (1-bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)
    else:
        graph = torch.sigmoid(sampling_weights)
    return graph
