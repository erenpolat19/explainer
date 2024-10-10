from model.models import *
from data_utils.data_preprocessing import *
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=1, help='k')
parser.add_argument
args = parser.parse_args()

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.  
        #print(data)          
        data.to(device)
        #print(data.edge_index.shape)
        out = model(data.x, data.edge_index, edge_weights = None, batch = data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        train_loss = train_loss + loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    return train_loss / len(train_loader)

def test(loader, model, device):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            data.to(device)
            with torch.no_grad():
                out = model(data.x, data.edge_index, edge_weights = None, batch = data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                y = data.y
                correct += int((pred == y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    device = 'cpu'
    #dataset_name = 'BA-2motif'
    dataset_name = 'BA-2motif-this-one-works'
    data = preprocess_ba_2motifs(dataset_name)
    #data = preprocess_generated_ba2()
    train_loader, val_loader, test_loader = get_dataloaders(data, args, batch_size=64, val_split=0.1, test_split=0.1)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    num_node_features = data[0].x.shape[1]
    #print(data[1])

    model = GCN(num_node_features,2, 'both').to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None
    best_epoch = 0
    for epoch in range(1, 1000):
        train_loss = train(model, criterion, optimizer, train_loader, device)
        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        if best_val_acc < val_acc:
            best_model = model
            best_val_acc = val_acc
            best_epoch = epoch
        #test_acc = test(test_loader, model, data, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    print('Final test' , test(test_loader, model, device), f'best epoch {best_epoch}')
    

    torch.save(model.state_dict(), 'clf-good-both.pth')