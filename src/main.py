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

import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

def extract_individual_graphs(batch, expl_mask):
    # Get the number of graphs in the batch
    num_graphs = batch.batch.max().item() + 1

    graphs = []
    expl_masks_split = [] 
    
    # Iterate over each graph in the batch
    for graph_idx in range(num_graphs):
        # Find nodes that belong to the current graph
        node_mask = batch.batch == graph_idx
        
        # Get node indices for this graph
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)
        
        # Extract node features for this graph
        node_features = batch.x[node_indices]
        
        # Filter edges for the current graph
        edge_mask = node_mask[batch.edge_index[0]]  # Only keep edges where the source node belongs to the current graph
        edge_index = batch.edge_index[:, edge_mask]
        
        # Remap the node indices for the edge_index (so indices are zero-based for each graph)
        edge_index_remapped = edge_index.clone()
        node_id_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Apply remapping safely with default if the node is missing
        edge_index_remapped = edge_index_remapped.apply_(lambda idx: node_id_map.get(idx, -1))

        # print(f"Graph {graph_idx}: node_indices = {node_indices.tolist()}")
        # print(f"Graph {graph_idx}: edge_index (before remapping) = {edge_index}")
        # print(f"Graph {graph_idx}: edge_index_remapped (after remapping) = {edge_index_remapped}")

        # Remove edges with unmapped nodes (which were not part of the node_id_map)
        valid_edges_mask = (edge_index_remapped[0] != -1) & (edge_index_remapped[1] != -1)
        edge_index_remapped = edge_index_remapped[:, valid_edges_mask]

        # Filter expl_mask for the current graph's edges
        expl_mask_split = expl_mask[edge_mask][valid_edges_mask]
        expl_masks_split.append(expl_mask_split)

        # Create a dictionary for the current graph
        graph_data = {
            'nodes': node_features,
            'edges': edge_index_remapped
        }
        
        graphs.append(graph_data)

    return graphs, expl_masks_split

def visualize(graphs, expl_masks_split, top_k):
    for i, graph_data in enumerate(graphs):
        # Create a NetworkX graph
        G = nx.Graph()

        # Add nodes
        num_nodes = graph_data['nodes'].shape[0]
        G.add_nodes_from(range(num_nodes))

        # Add edges
        edges = graph_data['edges'].t().tolist()
        G.add_edges_from(edges)

        # Set up the plot
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G) 
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightblue", alpha=0.9)

        # Draw edges
        edge_colors = 'black'
        expl_mask = expl_masks_split[i]  # Use the explanation mask corresponding to this graph
        sorted_indices = expl_mask.argsort(descending=True)
        top_k_indices = sorted_indices[:top_k]  # Get indices of top k edges
        top_k_edges = [edges[idx] for idx in top_k_indices]
        top_k_mask = expl_mask[top_k_indices]  # Corresponding importance values

        non_top_k_indices = sorted_indices[top_k:]
        non_top_k_edges = [edges[idx] for idx in non_top_k_indices]

        # Add the top k edges to the graph
        G.add_edges_from(top_k_edges)
        edge_colors = plt.cm.Reds(top_k_mask) 
        nx.draw_networkx_edges(G, pos, edgelist=non_top_k_edges, edge_color='black')
        nx.draw_networkx_edges(G, pos, edgelist=top_k_edges, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
        
        # Title
        plt.title(f'Graph {i+1}', fontsize=16)
        plt.axis('off')  

        # Show the plot
        plt.show()
        

from sklearn.metrics import roc_auc_score


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

sys.path.append('../')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='dimension of z') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--h_dim', type=int, default=20, metavar='N', help='dimension of h') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='BA-2motif', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=1e-3, #changed to 3e-3
                    help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

parser.add_argument('--experiment_type', default='train', choices=['train', 'test', 'baseline'],
                    help='train: train CLEAR model; test: load CLEAR from file; baseline: run a baseline')
parser.add_argument('--reg_coefs', default=(0.05, 1.0), help='reg coefs')
parser.add_argument
args = parser.parse_args()

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def loss_f(pred, target, mask, reg_coefs):
    
    scale = 0.99
    mask = mask*(2*scale-1.0)+(1.0-scale)
    
    cce_loss = F.cross_entropy(pred, target)
    size_loss = torch.sum(mask) * reg_coefs[0]
    mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    mask_ent_loss = reg_coefs[1] * torch.mean(mask_ent_reg)
    
    return cce_loss + size_loss + mask_ent_loss

def eval_acc(clf_model, expl_model, dataloader, device, args, vis=False):
    expl_model.eval()
    correct = 0
    for data in dataloader:  # Iterate in batches over the training/test dataset.
        x, edge_index, y_target = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        
        with torch.no_grad():
<<<<<<< HEAD
            inputs = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            x, edge_index, y_target = inputs
            #edge_label = data.edge_label.to(device)
            expl_mask = explain_inference(clf_model, expl_model, inputs)
=======

            node_emb = clf_model.embedding(x, edge_index) # num_nodes x h_dim
            edge_emb = create_edge_embed(node_emb, edge_index) # E x 2*h_dim
            sampling_weights = expl_model(edge_emb)
            expl_mask = sample_graph(sampling_weights, bias=0.0, training=False).squeeze()
            if not vis:
                graphs, expl_masks_split = extract_individual_graphs(data, expl_mask)
                visualize(graphs, expl_masks_split, top_k=10)
                vis = True
>>>>>>> 0d67dd451fd9e7712ee4088fc1df824cab4161c7
            # Using the masked graph's edge weights
            masked_pred = clf_model(x, edge_index, edge_weights = expl_mask, batch=data.batch)   # Graph-level prediction
            y_pred = masked_pred.argmax(dim=1)
            correct += int((y_pred == y_target).sum())

    print('mask:', expl_mask.sum(), 'graph:', edge_index[0].shape)
    
    return correct / len(dataloader.dataset)

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

def train(clf_model, factual_explainer, optimizer_f, train_loader, val_loader, test_loader, device, args, temp=(5.0, 2.0)):
    temp_schedule = lambda e: temp[0] * ((temp[1] / temp[0]) ** (e / args.epochs))
    for epoch in range(args.epochs):
        factual_explainer.train()
        t = temp_schedule(epoch)
        total_loss_f = 0

        
#         for batch in train_loader:
#             x, edge_index, y_target = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)


        for data in train_loader:
            x, edge_index, y_target = data.x.to(device), data.edge_index.to(device), data.y.to(device)

            with torch.no_grad():
                node_emb = clf_model.embedding(x, edge_index) # num_nodes x h_dim
            
            edge_emb = create_edge_embed(node_emb, edge_index) # E x 2*h_dim

            sampling_weights = factual_explainer(edge_emb)
            
            expl_mask = sample_graph(sampling_weights, t, bias=0.0).squeeze()
            masked_pred = clf_model(x, edge_index, expl_mask, data.batch)  # Graph-level prediction

            optimizer_f.zero_grad()
  
            # Loss for factual explainer
            # loss_f = KL div + clf loss
            reg_coefs = args.reg_coefs
            reg_coefs = (1e-3, reg_coefs[1]) # tuned
            loss = loss_f(masked_pred, y_target, expl_mask, reg_coefs)

            loss.backward()
            optimizer_f.step()

            total_loss_f += loss.item()

        val_acc = eval_acc(clf_model, factual_explainer, val_loader, device, args)
        train_acc = eval_acc(clf_model, factual_explainer, val_loader, device, args)
        print(f"Epoch {epoch + 1}/{args.epochs}, Factual Loss: {loss}, Val_acc: {val_acc},  Train_acc: {train_acc}")

    test_acc = eval_acc(clf_model, factual_explainer, test_loader, device, args, vis=True)
    print(f"Final Test_acc: {test_acc}")

    roc_auc = eval_explain(clf_model, factual_explainer, test_loader, device)
    print('Final test ROC AUC:', roc_auc)

def run(args):
    dataset_name = args.dataset
    device = "cpu"
    """
    load data for train, val, test
    """
    data = preprocess_ba_2motifs(dataset_name, padded=False)
    # with open('../dataset/BA-2motif/raw/ba2-generated.pkl', 'rb') as f:
    #     data = pkl.load(f)
        
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.1, test_split=0.1)

    """
    All we are doing now is: Batched graphs --> GCN to get this pretrained classifier's node embeddings
    --> make these into edge embeddings, z_uv (E num edges x 2*h_dim) --> FactualExplainer MLP --> sample & get factual mask (E x 1) 
    """
    params = {}
    # params
    params['x_dim'] = 10
    params['num_classes'] = 2
    # embedder
    clf_model = GCN(params['x_dim'], params['num_classes']).to(device)              # load best model
    
    # Load the saved state dictionary

    #checkpoint = torch.load('clf.pth')
    checkpoint = torch.load('clf2.pth')
    #checkpoint = torch.load('clf-generated.pth')

    # Load the weights into the model
    clf_model.load_state_dict(checkpoint)
    clf_model.eval()                                                              

    # Factual Explainer MLP
    expl_embedding = args.h_dim * 2
    factual_explainer = FactualExplainer(expl_embedding, device)
    optimizer_f = Adam(factual_explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    train(clf_model, factual_explainer, optimizer_f, train_loader, val_loader, test_loader, device, args)
    
run(args)






'''1. dataloader for mutag
        2. pretrain the graph classifier
        3. initialize factual and counterfactual models
            (use the same encoder network w 2 decoders)
        4. for each batch:
            forward pass, encode graph G to z_mu, z_logvar
            sample z with reparam. trick
            decode explanations a_f, a_cf
            encode a_cf to z_cf_mu, z_cf_logvar
            sample z_cf
            decode a_cf_f
            we have a_f, a_cf, a_cf_f

            loss = d(a_f, a_cf_f), I(y=1, a_cf_f), I(y=0, a_f), I(y=1, a_cf)??,
                KL(z_u_logvar - prior_z_logvar), KL(a, a_cf), 

            check proxy graph paper potentially for k_1 iterations on one loss etc., and k_2 on the other

            check clear for setting some loss terms to 0 in the beginning e_0 epochs

            backpropagate

                -add something about the validation set? test our metrics on data_val
                -possibly add perturbation stuff later?
                -how to make sure of the causal graph relevant w counterfactual
                -maybe construct a dataset or find a simple observable one (smth like shapes)
                -look at how to make things global, anchors/prototypes? rkhs (ugh)?, 
                                            unsup clustering a start? extract global for locals?'''
    


'''
    g, y_g
    y_cf = !y_g
    encoder = Encoder(x_dim , h_dim, z_dim)

    factExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)
    cfExplainer = Explainer(encoder, z_dim, a_out_size, x_out_size)

    factual_exp = factExplainer(x, edge_index, edge_weight, y_g)
    cf_exp = cfExplainer(x, edge_index, edge_weight, y_cf)
'''