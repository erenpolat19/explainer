import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
import random

import pickle as pkl

dataset1 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=20, num_edges=1),
    motif_generator=HouseMotif(),
    num_motifs=1,
    num_graphs=500,
)

dataset2 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=20, num_edges=1),
    motif_generator=CycleMotif(5),
    num_motifs=1,
    num_graphs=500,
)

graphs = []
for data in dataset1:
    graphs.append(Data(x=0.1000 * torch.ones((25,10), dtype=torch.float), edge_index=data.edge_index, edge_label=data.edge_mask,
                             y=torch.zeros((1,), dtype=torch.long)))

for data in dataset2:
    graphs.append(Data(x=0.1000 * torch.ones((25,10), dtype=torch.float), edge_index=data.edge_index, edge_label=data.edge_mask,
                             y=torch.zeros((1,), dtype=torch.long)))

random.shuffle(graphs)
# Save the list to a file
with open('ba2-generated.pkl', 'wb') as f:
    pkl.dump(graphs, f)

