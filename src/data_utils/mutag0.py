"""Read the Mutag dataset and create the graphx"""

import numpy as np
import os
import dgl
from dgl.data import DGLDataset
import torch
from dgl import save_graphs, load_graphs
from utils.common_utils import read_file


class MutagDataset0(DGLDataset):
    def __init__(self, edges=None, graph_indicator=None, node_labels=None, edge_labels=None, graph_labels=None, load_path=None):
        super().__init__(name='mutag0')
        if load_path:
            self.load_path = load_path
            self.load_()
        else:
            

    def process(self):
        print('processing')

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def save_(self, save_path):
        save_graphs(save_path, self.graphs, {'labels': self.labels})

    def load_(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(os.path.join(self.load_path, 'dgl_graph.bin'))
        self.labels = label_dict['labels']
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]


def mutag_preprocessing_0(dataset_dir):
    name = "Mutagenicity"
    edge_path = os.path.join(dataset_dir, name + "_A.txt")
    edge_label_path = os.path.join(dataset_dir, name + "_edge_gt.txt")
    graph_indicator_path = os.path.join(dataset_dir, name + "_graph_indicator.txt")
    node_label_path = os.path.join(dataset_dir, name + "_node_labels.txt")
    graph_label_path = os.path.join(dataset_dir, name + "_graph_labels.txt")
    edge_data = read_file(edge_path)
    edge_data = np.array(edge_data)
    edge_data = edge_data - 1
    edge_labels = read_file(edge_label_path)
    edge_labels = np.array(edge_labels)
    graph_indicator = read_file(graph_indicator_path) - 1
    node_labels = read_file(node_label_path)
    graph_labels = read_file((graph_label_path))
    G_dataset = MutagDataset0(edge_data, graph_indicator, node_labels, edge_labels, graph_labels)
    return G_dataset

