import torch
import torch.nn as nn
from torch.nn import ReLU, Linear
from numbers import Number
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, global_add_pool, GraphConv
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
            out = torch.cat([out1, out2], dim=-1)     
        return out
    
    def forward(self, x, edge_index, edge_weights, y_target, batch): 
        graph_emb = self.graph_pooling(self.embedder.embedding(x, edge_index, edge_weights=edge_weights), batch, type='mean')
        
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
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2 + 1),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2 + 1, output_size)
        )

        self.fc_mu = nn.Linear(256, hidden_dim)
        self.fc_logvar = nn.Linear(256, hidden_dim)

    def encode(self, inputs, y_target):
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

    def forward(self,inputs,beta, y_target, batch=None):
        
        mu, logvar = self.encode(inputs, y_target)
        z = self.reparameterize(mu, beta * logvar)
        print('z', z.shape)
        if batch is None:
            out1, _ = torch.max(z, 0)
            out1 = out1.unsqueeze(0)
            out2 = torch.mean(z, 0).unsqueeze(0)
        else:
            out1 = global_max_pool(z, batch)
            out2 = global_mean_pool(z, batch)
        print('out1', out1.shape)
        
        reduce_z = torch.cat([out1, out2], dim=-1)
        reduce_z = torch.cat([reduce_z, y_target.unsqueeze(-1)], dim=-1)
        recon_x = self.decode(reduce_z)

        return recon_x, mu, logvar
    

#CLEAR MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
    

 #Adapted from clear   
class GraphCFE(nn.Module):
    def __init__(self, init_params, args):
        super(GraphCFE, self).__init__()
        self.vae_type = init_params['vae_type']  # graphVAE
        self.x_dim = init_params['x_dim']
        self.h_dim = args.dim_h
        self.z_dim = args.dim_z
        self.u_dim = 1 # init_params['u_dim']
        self.dropout = args.dropout
        self.max_num_nodes = init_params['max_num_nodes']
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.disable_u = args.disable_u

        if self.disable_u:
            self.u_dim = 0
            print('disable u!')
        if self.encoder_type == 'gcn':
            self.graph_model = GCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = GraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device)
        self.prior_var = nn.Sequential(MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device), nn.Sigmoid())

        # encoder
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

        # decoder
        self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())
        self.graph_norm = nn.BatchNorm1d(self.h_dim)

    def encoder(self, features, u, adj, y_cf):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        #graph_rep = self.graph_norm(graph_rep)

        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))

        return z_mu, z_logvar

    def get_represent(self, features, u, adj, y_cf):
        u_onehot = u
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf, u):
        if self.disable_u:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                              self.max_num_nodes)
        else:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.max_num_nodes)

        features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.x_dim)
        return features_reconst, adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u):  # P(Z|U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u),self.h_dim)).to(device)
            z_u_logvar = torch.ones((len(u),self.h_dim)).to(device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, u, adj, y_cf):
        u_onehot = u

        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)

        return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu, 'z_u_logvar': z_u_logvar}