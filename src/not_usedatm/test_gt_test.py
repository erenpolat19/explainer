from model.models import *
from utils import *

sys.path.append('../')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

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
parser.add_argument('--reg_coefs', default=(1e-3, 1.0), help='reg coefs')
parser.add_argument('--k', default=10, help='k')
parser.add_argument
args = parser.parse_args()


dataset_name = 'BA-2motif-this-one-works'
#dataset = Mutag(root=data_dir / 'mutag')

data = preprocess_ba_2motifs(dataset_name)
train_loader, val_loader, test_loader = get_dataloaders(data, args, batch_size=args.batch_size, val_split=0.1, test_split=0.1)

"""
All we are doing now is: Batched graphs --> GCN to get this pretrained classifier's node embeddings
--> make these into edge embeddings, z_uv (E num edges x 2*h_dim) --> FactualExplainer MLP --> sample & get factual mask (E x 1) 
"""

params = {}
params['x_dim'] = 10
params['num_classes'] = 2

# embedder
clf_model = GCN(10, 2).to('cpu')              # load best model

# Load the saved state dictionary
checkpoint = torch.load('pretrained/best_model')

# Rename the keys in the checkpoint to match your current model
new_state_dict = {}
for key in checkpoint['model_state_dict']:
    new_key = key.replace('.weight', '.lin.weight')  # Adjust this based on your needs
    new_state_dict[new_key] = checkpoint['model_state_dict'][key]

# Load the modified state dictionary
clf_model.load_state_dict(new_state_dict)

# Load the weights into the mod
clf_model.eval()

train_loader, val_loader, test_loader = get_dataloaders(data, args, batch_size=args.batch_size, val_split=0.1, test_split=0.1)
device = 'cpu'

correct= 0
for data in test_loader:  # Iterate in batches over the training/test dataset.
    x, edge_index, y_target, edge_label = data.x.to(device), data.edge_index.to(device), data.y.to(device), data.edge_label
    inputs =  x, edge_index, y_target
    # Using the masked graph's edge weights
    masked_pred = clf_model(x, edge_index, edge_weights = edge_label, batch=data.batch)   # Graph-level prediction
    y_pred = masked_pred.argmax(dim=1)
    correct += int((y_pred == y_target).sum())
    

print(correct / len(test_loader.dataset))