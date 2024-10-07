from data_preprocessing import *
from visualize import *

dataset_name= 'BA-2motif'
data = preprocess_ba_2motifs(dataset_name)
train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64, val_split=0.1, test_split=0.1)
device = 'cpu'

for data in test_loader:
    x, edge_index, y_target, edge_label = data.x.to(device), data.edge_index.to(device), data.y.to(device), data.edge_label.to(device)
    graphs, expl_masks_split = extract_individual_graphs(data, edge_label)
    visualize(graphs, expl_masks_split, top_k=10)