
import numpy as np
import torch
import pickle as pkl
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data
import scipy.sparse as sp
from torch_geometric.utils import dense_to_sparse
from dataloader import get_dataloaders

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to an edge index
    :param adj: Original adjacency matrix
    :return: Edge index representation of the graphs
    """
    converted = []
    for d in adj:
        edge_index = np.argwhere(d > 0.).T
        mask = edge_index[0] != edge_index[1]
        converted.append(edge_index[:, mask])

    return converted

"""
Loads the BA-2motifs dataset from a pickle file.

Args:
    dataset (str): Name of the dataset to be loaded. 

Returns:
    (adjs, features, labels) (tuple): A tuple containing three elements:
        - adjs (list of numpy.ndarray): A list of adjacency matrices for each graph in the dataset.
        - features (list of numpy.ndarray): A list of feature matrices corresponding to each graph.
        - labels (list of numpy.ndarray): A list of labels associated with each graph.
"""
def load_ba_2motifs(dataset):
    # with open('../dataset/BA-2motif/raw/BA-2motif.pkl', 'rb') as fin:
    #     adjs, features, labels = pkl.load(fin)
    with open('../dataset/BA-2motif/raw/BA-2motif-new.pkl', 'rb') as fin:
        adjs, features, labels = pkl.load(fin)
    return adjs, features, labels

"""
Preprocess the BA-2motifs dataset.

Args: 
    dataset (str): Name of the dataset to be loaded. 
    padded (bool): Set to True --> padding, False --> no padding.
    save_falg (bool): Saves data if flag is set.

Returns:
    data (Pytorch Geometric Data object): Preprocessed data, also saved to .pt file.
"""
def preprocess_ba_2motifs(dataset, padded=False, save_flag=True):
    dense_edges, node_features, graph_labels = load_ba_2motifs(dataset)
    #adjs, features, labels = load_ba_2motifs(dataset)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        x = torch.from_numpy(node_features[graph_idx]).float()
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        y = torch.from_numpy(np.where(graph_labels[graph_idx])[0]).reshape(-1, 1).long()

        node_label = torch.zeros(x.shape[0]).long()
        node_label[20:] = 1
        edge_label = ((edge_index[0] >= 20) & (edge_index[0] < 25) & (edge_index[1] >= 20) & (edge_index[1] < 25)).long()
        y = y.squeeze()
        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label))

    return data_list

    
    #Define max number of nodes
    max_num_nodes = 30
    
    adj_all = []            # List to store adjacency matrices
    features_all = []       # List to store feature matrices
    labels_all = []         # List to store labels
    edge_indices_all = []
    
    for i in range(len(adjs)):
        adj = adjs[i]
        feature = features[i]
        label = labels[i]

        adj_all.append(adj)
        features_all.append(feature)
        labels_all.append(label)

        # # Skip graphs with more than max num nodes
        # if adj.shape[0] > max_num_nodes:
        #     continue
        
        # if padded:
        #     # Pad the adjacency matrix
        #     if adj.shape[0] < max_num_nodes:
        #         padded_adj = np.zeros((max_num_nodes, max_num_nodes))
        #         padded_adj[:adj.shape[0], :adj.shape[1]] = adj
        #         adj = padded_adj
        
        #     # Pad the feature matrix
        #     if feature.shape[0] < max_num_nodes:
        #         padded_features = np.zeros((max_num_nodes, feature.shape[1]))
        #         padded_features[:feature.shape[0], :] = feature
        #         feature = padded_features
        
        # # Convert adjacency matrix to edge_index
        # adj_sparse = sp.coo_matrix(adj)  
        # edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
        
        # # Create PyTorch Geometric Data object
        # data = Data(x=torch.tensor(feature, dtype=torch.float), 
        #             edge_index=edge_index,
        #             y=torch.tensor(label, dtype=torch.float))
    
    edge_indices_all = adj_to_edge_index(adjs)
    
    # Save the processed data
    if save_flag:
        path_save = '../dataset/BA-2motif/processed/ba2motifs.pt'
        torch.save({'data': [Data(x=torch.tensor(features_all[i], dtype=torch.float), 
                                  edge_index=torch.tensor(edge_indices_all[i], dtype=torch.long),
                                  y=torch.tensor(labels_all[i], dtype=torch.float).argmax())
                              for i in range(len(adjs))]}, path_save)
        print('Saved data:', path_save)

    

    return [Data(x=torch.tensor(features_all[i], dtype=torch.float), 
                 edge_index=torch.tensor(edge_indices_all[i], dtype=torch.long),
                 y=torch.tensor(labels_all[i], dtype=torch.float).argmax()) 
            for i in range(len(adjs))]

if __name__ == '__main__':
    dataset = 'ba2motifs'
    data = preprocess_ba_2motifs(dataset, padded=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)
    

