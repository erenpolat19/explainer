import torch
import torch.nn as nn
from torch.nn import ReLU, Linear
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.h_dim = 20
        self.conv1 = GCNConv(num_features, self.h_dim)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(self.h_dim, self.h_dim)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(self.h_dim, self.h_dim)
        self.relu3 = ReLU()
        self.lin = Linear(self.h_dim, num_classes)

    def forward(self, x, edge_index, edge_weights=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out = self.lin(out1)
        return out
    
    def reset_parameters(self):
        # Initialize weights using Xavier initialization
        for layer in self.children():
            if isinstance(layer, GCNConv):
                # Xavier initialization for GCNConv weights
                for param in layer.parameters():
                    if param.dim() == 1:  # Bias
                        init.zeros_(param)
                    elif param.dim() == 2:  # Weight
                        init.xavier_uniform_(param)


    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
            #edge_weights = torch.ones(edge_index.size(1)).cuda()

        stack = []
        x = x.float()
        out1 = self.conv1(x, edge_index, edge_weights)              # conv1: Input dim = num_features, Output dim = 20

        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)                                         

        out2 = self.conv2(out1, edge_index, edge_weights)           # conv2: Input dim = 20, Output dim = 20
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)           # conv3: Input dim = 20, Output dim = 20
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3                                           # lin: Input dim = 20, Output dim = num_classes

        return input_lin #num_nodes * h_dim



