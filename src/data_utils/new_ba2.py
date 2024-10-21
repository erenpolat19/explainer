from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
import os.path as osp
import pickle

def read_ba2motif_data(folder: str, prefix):
    with open(osp.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        x = torch.from_numpy(node_features[graph_idx]).float()
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        y = torch.from_numpy(np.where(graph_labels[graph_idx])[0]).reshape(-1, 1).float()

        node_label = torch.zeros(x.shape[0]).float()
        node_label[20:] = 1
        edge_label = ((edge_index[0] >= 20) & (edge_index[0] < 25) & (edge_index[1] >= 20) & (edge_index[1] < 25)).float()

        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label))
    return data_list

class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])