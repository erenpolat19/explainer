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


def explain_inference(clf_model, expl_model, inputs, device='cpu', bias = 0.0):
    with torch.no_grad():
        x, edge_index, y_target = inputs
        node_emb = clf_model.embedding(x, edge_index) # num_nodes x h_dim
        edge_emb = create_edge_embed(node_emb, edge_index) # E x 2*h_dim
        sampling_weights = expl_model(edge_emb)
        expl_mask = sample_graph(sampling_weights, bias=bias, training=False).squeeze()
    
    return expl_mask

def eval_explain(clf_model, expl_model, dataloader, device='cpu'):
    expl_model.eval()
    predictions = []
    ground_explanations = []
    for data in dataloader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            inputs = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            x, edge_index, y_target = inputs
            edge_label = data.edge_label.to(device)
            expl_mask = explain_inference(clf_model, expl_model, inputs)

            assert expl_mask.shape == edge_label.shape

            for idx in range(expl_mask.shape[0]):
                predictions.append(expl_mask[idx].item())
                ground_explanations.append(edge_label[idx].item())
            
    return roc_auc_score(ground_explanations, predictions)

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