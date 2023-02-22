from forward_backward_pass import forward_backward_pass
from data import get_data_mnist
from data import get_data_cifar10
from lenet5_1d import LeNet5_1D
from lenet5_3d import LeNet5_3D
from params import DEVICE, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, PATIENCE

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from copy import deepcopy


def train_lenet(dataset_name: str) -> Module:
    # If dataset_name is 'mnist', then we will train the model on the MNIST dataset.
    # If dataset_name is 'cifar10', then we will train the model on the CIFAR-10 dataset.
    if dataset_name == 'mnist':
        train_loader, val_loader, _ = get_data_mnist(BATCH_SIZE)
        model = LeNet5_1D()
    elif dataset_name == 'cifar10':
        train_loader, val_loader, _ = get_data_cifar10(BATCH_SIZE)
        model = LeNet5_3D()

    # Move the model to the appropriate device
    model = model.to(DEVICE)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Criterion for early stopping
    best_val_loss = 10e10
    best_epoch = 0
    best_model = deepcopy(model)

    # Train the model
    for epoch in range(N_EPOCHS):
        # Train
        model, train_loss, train_acc = forward_backward_pass(
            model, optimizer, train_loader, DEVICE)

        # Validation
        model, val_loss, val_acc = forward_backward_pass(
            model, None, val_loader, DEVICE)
        
        print(f'Epoch: {epoch:02} | Epoch Train Loss: {train_loss:.3f} | Epoch Train Acc: {train_acc * 100:.2f}%')
        
        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
        elif epoch - best_epoch > PATIENCE:
            print('Early stopping!. Best epoch: ', best_epoch)
            break

    # Save the model
    torch.save(best_model.state_dict(), f'models/lenet5/lenet5_{dataset_name}.pt')

    return model