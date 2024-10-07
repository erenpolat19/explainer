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
from visualize import *

from sklearn.metrics import roc_auc_score
import os 

# USE ONLY FOR INFERENCE
def explain_inference(clf_model, expl_model, inputs, device='cpu', bias = 0.0):
    with torch.no_grad():
        x, edge_index, y_target = inputs
        node_emb = clf_model.embedding(x, edge_index) # num_nodes x h_dim
        edge_emb = create_edge_embed(node_emb, edge_index) # E x 2*h_dim
        sampling_weights = expl_model(edge_emb)
        expl_mask = sample_graph(sampling_weights, bias=bias, training=False).squeeze()
    
    return expl_mask

def eval_acc(clf_model, expl_model, dataloader, device, args, v=False):
    expl_model.eval()
    correct = 0
    vis = False
    for data in dataloader:  # Iterate in batches over the training/test dataset.
        x, edge_index, y_target = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        
        with torch.no_grad():
            node_emb = clf_model.embedding(x, edge_index) # num_nodes x h_dim
            edge_emb = create_edge_embed(node_emb, edge_index) # E x 2*h_dim
            sampling_weights = expl_model(edge_emb)
            expl_mask = sample_graph(sampling_weights, bias=0.0, training=False).squeeze()
            if not vis and v:
                graphs, expl_masks_split = extract_individual_graphs(data, expl_mask)
                visualize(graphs, expl_masks_split, top_k=10)
                vis = True
            # Using the masked graph's edge weights
            masked_pred = clf_model(x, edge_index, edge_weights = expl_mask, batch=data.batch)   # Graph-level prediction
            y_pred = masked_pred.argmax(dim=1)
            correct += int((y_pred == y_target).sum())
            
        print('mask:', expl_mask.sum(), 'graph:', edge_index[0].shape)
    return correct / len(dataloader.dataset)
    

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