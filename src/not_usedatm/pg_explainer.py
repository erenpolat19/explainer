from model.pg_model import *
from model.gcn import *
import sys
import argparse
from data_utils.data_preprocessing import *
from sklearn.metrics import roc_auc_score
from torch_geometric.explain import Explainer, ModelConfig, ThresholdConfig
from utils import *


sys.path.append('../')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='dimension of z') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--h_dim', type=int, default=20, metavar='N', help='dimension of h') #I AM CHANGING FROM 16 TO 20 FOR DEBUG
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='BA-2motif', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])

parser.add_argument
args = parser.parse_args()

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def eval_acc_pg(explainer, dataloader, clf_model, device='cpu'):
    correct = 0
    clf_model.eval()
    for data in dataloader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():

            x, edge_index, y_target, edge_label = data.x.to(device), data.edge_index.to(device), data.y.to(device), data.edge_label.to(device)
            expl_mask = explainer(x, edge_index, target=y_target).edge_mask
            masked_pred = clf_model(x, edge_index, edge_weights = expl_mask, batch=data.batch)   # Graph-level prediction
            
            correct += int((masked_pred.argmax(dim=1) == y_target.argmax()).sum())
            
    return correct / len(dataloader.dataset)

def eval_exp_pg(explainer, dataloader, device='cpu'):
    predictions = []
    ground_explanations = []

    for data in dataloader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            x, edge_index, y_target, edge_label = data.x.to(device), data.edge_index.to(device), data.y.to(device), data.edge_label.to(device)
            expl_mask = explainer(x, edge_index, target=y_target).edge_mask
            assert expl_mask.shape == edge_label.shape

            for idx in range(expl_mask.shape[0]):
                predictions.append(expl_mask[idx].item())
                ground_explanations.append(edge_label[idx].item())
    return roc_auc_score(ground_explanations, predictions)

def train(clf_model, explainer, train_loader, val_loader, test_loader, device, args, temp=(5.0, 2.0)):
    for epoch in range(args.epochs):
        for data in train_loader:
            x, edge_index, y_target = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            loss = explainer.algorithm.train(epoch, clf_model, x, edge_index,
                                            target=y_target)

        print(f"Epoch {epoch + 1}/{args.epochs}, Factual Loss: {loss}")
    print( 'Test ROC AUC', eval_exp_pg(explainer, test_loader, device))
    print( 'Test acc', eval_acc_pg(explainer, test_loader, clf_model, device))

def explain_and_save(explainer, data, device):
    res = []
    for graph in data:
        x, edge_index, y_target = graph.x.to(device), graph.edge_index.to(device), graph.y.to(device)
        res.append((graph, explainer(x, edge_index, target=y_target)))

    # with open('pg_w_explanations.pkl', 'wb') as f:
    #     pkl.dump(res, f)
    
    # Save the entire PGExplainer object using pickle
    with open('pretrained/pg_pretrained.pkl', 'wb') as f:
        pkl.dump(explainer, f)


def run(args):
    device = "cpu"
    """
    load data for train, val, test
    """
    #dataset_name = args.dataset
    dataset_name = 'BA-2motif-this-one-works'
    data = preprocess_ba_2motifs(dataset_name)
    for i, graph in enumerate(data):
        if graph.y.item() == 1:
            graph.y = torch.tensor([0,1], dtype=torch.float)
        else:
            graph.y = torch.tensor([1,0], dtype=torch.float)

    train_loader, val_loader, test_loader = get_dataloaders(data, args, batch_size=1, val_split=0.1, test_split=0.1)
    params = {}
    # params
    params['x_dim'] = 10
    params['num_classes'] = 2

    # classifier
    clf_model = GCN(params['x_dim'], params['num_classes'], 'both').to(device)              # load best model
    checkpoint = torch.load('pretrained/clf-good.pth')
    clf_model.load_state_dict(checkpoint)
    clf_model.eval()                                                              

    # Factual Explainer MLP
    explainer = Explainer(
        model= clf_model,
        algorithm=PGExplainer(epochs=args.epochs, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=ModelConfig(mode='binary_classification', task_level='graph', return_type='raw'))
    
    train(clf_model, explainer, train_loader, val_loader, test_loader, device, args)
    explain_and_save(explainer, data, device)
    
run(args)
