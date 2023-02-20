from data import get_data_mnist, get_data_cifar10
from params import BATCH_SIZE, DEVICE
from lenet5_1d import LeNet5_1D
from lenet5_3d import LeNet5_3D
from forward_backward_pass import forward_backward_pass

import torch


def test_lenet(dataset_name: str):
    # If dataset_name is 'mnist', then we will test the model on the MNIST dataset.
    # If dataset_name is 'cifar10', then we will test the model on the CIFAR-10 dataset.
    if dataset_name == 'mnist':
        _, _, test_loader = get_data_mnist(BATCH_SIZE)
        model = LeNet5_1D()
    elif dataset_name == 'cifar10':
        _, _, test_loader = get_data_cifar10(BATCH_SIZE)
        model = LeNet5_3D()
        
    # Load the model
    model.load_state_dict(torch.load(f'models/lenet5/lenet5_{dataset_name}.pt'))
    
    # Move the model to the appropriate device
    model = model.to(DEVICE)
    
    # Test
    model, test_loss, test_acc = forward_backward_pass(
        model, None, test_loader, DEVICE)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')