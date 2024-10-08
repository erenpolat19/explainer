import torch
import torch.nn as nn
from torch.nn import ReLU, Linear
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, global_add_pool
from model.gcn import *
import torch.nn.functional as F

class FactualExplainer(nn.Module):
    def __init__(self, expl_embedding, device):
        super(FactualExplainer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
    
    def forward(self, graph_emb): 
        return self.mlp(graph_emb)

class GraphEncoder(nn.Module):
    def __init__(self, embedder, x_dim, h_dim, z_dim):
        super(GraphEncoder, self).__init__()
        self.embedder = embedder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

    def graph_pooling(self, embed, batch, type='mean'):
        if type == 'max':
            out = global_max_pool(embed, batch)
        elif type == 'sum':
            out = global_add_pool(embed, batch)
        elif type == 'mean':
            out = global_mean_pool(embed, batch)
        elif type == 'both':
            out1 = global_max_pool(embed, batch)
            out2 = global_mean_pool(embed, batch)
        return out
    
    def forward(self, x, edge_index, edge_weights, y_target, batch): 
        graph_emb = self.graph_pooling(self.embedder.embedding(x, edge_index, edge_weights=edge_weights), )
        
        z_mu = self.encoder_mean(torch.cat((graph_emb, y_target), dim=1))
        z_logvar = self.encoder_var(torch.cat((graph_emb, y_target), dim=1))

        return z_mu,z_logvar
    
class Decoder(nn.Module):
    def __init__(self, z_dim, output_size):
    
        self.decoder = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(),
            nn.Linear(z_dim * 2, output_size)
        )
    def forward(self, z):
        return torch.sigmoid(self.decoder(z))
    
class Explainer(nn.Module):
    def __init__(self, encoder, z_dim, a_out_size, x_out_size):
        self.encoder = encoder
        self.decoder_a = Decoder(z_dim, a_out_size)
        self.decoder_x = Decoder(z_dim, x_out_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def decode(self ,z_sample):
        return self.decoder_a(z_sample), self.decoder_x(z_sample)
    
    def forward(self, x, edge_index, edge_weight, y_target, beta=1):
        mu, logvar = self.encoder(x, edge_index, edge_weight, y_target)
        z_sample = self.reparameterize(mu, beta * logvar)

        recons_a, recons_x = self.decode(z_sample)

class GNN_MLP_VariationalAutoEncoder(nn.Module):

    def __init__(self, feature_size, output_size):
        super(GNN_MLP_VariationalAutoEncoder, self).__init__()
        self.conv1 = GCNConv(feature_size, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, 256)
        hidden_dim = 512
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_size)
        )

        self.fc_mu = nn.Linear(256, hidden_dim)
        self.fc_logvar = nn.Linear(256, hidden_dim)

    def encode(self, inputs):
        x, edge_index, edge_weight = inputs
    
        out1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index, edge_weight=edge_weight)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index, edge_weight=edge_weight)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)

        input_lin = out3

        mu = self.fc_mu(input_lin)
        logvar = self.fc_logvar(input_lin)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        scale = 1e-2
        eps = torch.randn_like(std) * scale
        return mu + eps * std
    
    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self,inputs,beta, batch=None):
        
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, beta * logvar)
        
        if batch is None:
            out1, _ = torch.max(z, 0)
            out1 = out1.unsqueeze(0)
            out2 = torch.mean(z, 0).unsqueeze(0)
        else:
            out1 = global_max_pool(z, batch)
            out2 = global_mean_pool(z, batch)

        reduce_z = torch.cat([out1, out2], dim=-1)
        recon_x = self.decode(reduce_z)

        return recon_x, mu, logvar