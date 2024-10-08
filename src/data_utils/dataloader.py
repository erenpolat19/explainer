import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

"""
Create train, validation, and test DataLoaders from the data list.

Args:
    data_list (list): List of PyTorch Geometric Data objects.
    batch_size (int): Batch size for the DataLoader.
    val_split (float): Fraction of data to be used for validation.
    test_split (float): Fraction of data to be used for testing.

Returns:
    train_loader (DataLoader): DataLoader for the training set.
    val_loader (DataLoader): DataLoader for the validation set.
    test_loader (DataLoader): DataLoader for the test set.
"""
def get_dataloaders(data_list, batch_size=32, val_split=0.2, test_split=0.2):

    num_samples = len(data_list)
    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=val_split + test_split, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_split / (val_split + test_split), shuffle=True)
    
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
