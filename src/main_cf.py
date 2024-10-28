from model.models import *
from utils import *

from torch_geometric.utils import to_dense_adj

sys.path.append('../')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--vae_cf_epoch', type=int, default=20, metavar='N',
                    help='epochs to train without cf loss')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='dimension of z') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--h_dim', type=int, default=20, metavar='N', help='dimension of h') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='BA-2motif', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=3e-3, #changed to 3e-3
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

def loss(reconstr_a, orig_a, orig_x, y_pred, y_cf, z_mu, z_logvar, alpha):
    #reconstr loss
    reconst_loss = F.binary_cross_entropy(reconstr_a, orig_a)

    size_loss = torch.sum(reconstr_a) * 0.0001

    #kl loss, maybe add the learned prior later etc.
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    
    # mask_ent_reg = -reconstr_a * torch.log(reconstr_a) - (1 - reconstr_a) * torch.log(1 - reconstr_a)
    # mask_ent_loss = 1 * torch.mean(mask_ent_reg)

    #cf loss
    cf_loss = F.cross_entropy(y_pred, y_cf)

    losses = {'kl_loss': kl_loss, 'cf_loss': cf_loss, 'reconstr_loss': reconst_loss}
    return reconst_loss + kl_loss + alpha * cf_loss + size_loss, reconst_loss, kl_loss, cf_loss

def loss_proxy(reconstr_a, orig_a, y_pred, y_cf, mu, logvar, alpha, reconstr_mask, recons_mask_nobeta):

    reconst_loss= F.binary_cross_entropy(reconstr_a, orig_a, reduction='mean')
    kl_loss = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)).mean()
    
    mse_loss = F.mse_loss(reconstr_mask, recons_mask_nobeta, reduction = "mean")
    cf_loss =  F.cross_entropy(y_pred, y_cf)

    return reconst_loss + kl_loss + alpha * cf_loss + mse_loss, reconst_loss, kl_loss, cf_loss


def get_counterfactual(inputs, cf_explainer, clf_model, data, params, y_cf, beta=1, device='cpu'):
    num_nodes = params['num_nodes']

    x, edge_index, edge_weights = inputs #x : batchsize x 25, 10   edge_index: 2x Einbatch
    #('x', x.shape, 'edge_index' , edge_index.shape)

    recons_mask_nobeta, _, _ = cf_explainer(inputs, beta=0, batch=data.batch)
    reconstr_mask, mu, logvar = cf_explainer(inputs, beta=1, batch=data.batch)

    # reconstr mask: batchsize, 25*25
    # x: batchsize*25  batchsize, 25
    batch_size = reconstr_mask.shape[0]
    orig_a = to_dense_adj(edge_index, data.batch, max_num_nodes=num_nodes)

    reconstr_adjs = []
    orig_adjs = []
    offset_new = 0
    aug_edge_list = []
    aug_edge_weights = []
        

    for i in range(batch_size):
        adj_matrix = reconstr_mask[i].reshape(num_nodes, num_nodes)
        orig_adjs.append(orig_a[i,:, :])
        reconstr_adjs.append(adj_matrix)
        edge_list = torch.nonzero(adj_matrix)
        edge_weights = adj_matrix[edge_list[:, 0], edge_list[:, 1]]

        edge_list = edge_list + offset_new
        aug_edge_list.append(edge_list.T)
        aug_edge_weights.append(edge_weights)
        offset_new += num_nodes

    aug_edge_list = torch.concat(aug_edge_list,-1).to(device) 
    aug_edge_weights = torch.concat(aug_edge_weights,-1).to(device)
    new_batch_tensor= torch.arange(batch_size).repeat_interleave(num_nodes)
    reconstr_a = torch.stack(reconstr_adjs).to(device)

    y_pred = clf_model(x, aug_edge_list, edge_weights=aug_edge_weights, batch=new_batch_tensor)

    assert(len(reconstr_adjs) == len(orig_adjs))
    all_cfs = list(zip(reconstr_adjs, orig_adjs))

    return reconstr_a, orig_a, y_pred, mu, logvar, recons_mask_nobeta, reconstr_mask, all_cfs


def test(cf_explainer, clf_model, loader, params, device, save=False, viz=False):
    cf_explainer.eval()
    total_loss_cf = 0
    correct = 0
    all_counterfactuals = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            x, edge_index, y_target  = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            y_cf = 1 - y_target
            reconstr_a, orig_a, y_pred, mu, logvar, recons_mask_nobeta, reconstr_mask, all_cfs = get_counterfactual((x, edge_index, None), cf_explainer, clf_model, data, params, y_cf, beta=1)
            # Using the masked graph's edge weights
            y_pred_hard = y_pred.argmax(dim=1)

            correct += int((y_pred_hard == y_cf).sum())
            alpha = 1
            loss_total, loss_reconstr, loss_kl, loss_cf = loss_proxy(reconstr_a, orig_a, y_pred, y_cf, mu, logvar, alpha, recons_mask_nobeta, reconstr_mask)

    if viz:      
        for data in loader:
            x, edge_index, y_target  = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            reconstr_a, orig_a, y_pred, mu, logvar, recons_mask_nobeta, reconstr_mask, all_cfs = get_counterfactual((x, edge_index, None), cf_explainer, clf_model, data, params, y_cf, beta=1)
            cf_binary = torch.bernoulli(reconstr_a)
            visualize_cfs(orig_a.squeeze(), cf_binary.squeeze())
            print(y_pred.argmax(dim=1))

    return correct / len(loader.dataset), loss_reconstr, loss_kl, loss_cf 

def train(clf_model, cf_explainer, optimizer_cf, train_loader, val_loader, test_loader, device, args, params, temp=(5.0, 2.0)):
    temp_schedule = lambda e: temp[0] * ((temp[1] / temp[0]) ** (e / args.epochs))
    for epoch in range(args.epochs):
        cf_explainer.train()
        t = temp_schedule(epoch)
        total_loss_cf = 0
        
        for data in train_loader:
            x, edge_index, y_target = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            y_cf = 1 - y_target
            optimizer_cf.zero_grad()

            #mask (or edge weight) is none atm
            reconstr_a, orig_a, y_pred, mu, logvar, recons_mask_nobeta, reconstr_mask, all_cfs = get_counterfactual((x, edge_index, None), cf_explainer, clf_model, data, params, y_cf, beta=1)
            alpha = 1

            if epoch < args.vae_cf_epoch: #check this later
                alpha = 0
            loss_total, loss_reconstr, loss_kl, loss_cf = loss_proxy(reconstr_a, orig_a, y_pred, y_cf, mu, logvar, alpha, recons_mask_nobeta, reconstr_mask)
            loss_total.backward()
            optimizer_cf.step()

            total_loss_cf += loss_total.item()

        cf_acc, test_reconstr, test_loss_kl, test_loss_cf = test(cf_explainer, clf_model, val_loader, params, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss_total.item()}, 'CF Acc': {cf_acc}, 'Recons loss: {test_reconstr}")


    cf_acc, test_reconstr, test_loss_kl, test_loss_cf = test(cf_explainer, clf_model, test_loader, params, device, viz=True)
    print(f"Final Test: Loss: {loss_total.item()}, 'CF Acc': {cf_acc}, 'Recons loss: {test_reconstr}")


def run(args):
    device = "cpu"
    """
    load data for train, val, test
    """
    #dataset_name = args.dataset
    dataset_name = 'BA-2motif-proxy'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, 'dataset')

    data = preprocess_ba_2motifs(data_dir, dataset_name)
    train_loader, val_loader, test_loader = get_dataloaders(data, args, batch_size=64, val_split=0.1, test_split=0.1)

    # x_dim = 10
    # h_dim = 20
    # z_dim = h_dim
    # num_classes = 2

    num_nodes = data[0].x.shape[0]
    params = {'x_dim': 10, 'h_dim': args.h_dim, 'z_dim': args.h_dim, 'num_classes': 2, 
              'num_nodes': num_nodes}

    clf_model = GCN(params['x_dim'], params['num_classes']).to(device)              # load clf
    checkpoint = torch.load(f'model/pretrained/ba2_proxy')
    clf_model.load_state_dict(checkpoint['model_state_dict'])
    clf_model.eval()                                                              

    #cf_explainer = GNN_MLP_VariationalAutoEncoder(params['x_dim'], params['num_nodes'] * params['num_nodes']).to(device)
    cf_explainer = VGAE(params['x_dim'], params['num_nodes'] * params['num_nodes'])

    optimizer_cf = Adam(cf_explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train(clf_model, cf_explainer, optimizer_cf, train_loader, val_loader, test_loader, device, args, params)

    val_loss = test(cf_explainer, clf_model, val_loader, params, device)
    test_loss = test(cf_explainer, clf_model, test_loader, params, device)

    print("Val loss: ", val_loss, "Test loss: ", test_loss)
    
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