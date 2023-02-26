import torchvision
from torchvision import transforms
import torch

def get_data_mnist(batch_size: int) -> tuple:
    # Transform the data by normaizing and transfering to tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32]), transforms.Normalize(0, 1)])

    # Load the data
    train_dataset = torchvision.datasets.MNIST(
        root='datasets/mnist', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='datasets/mnist', train=False, transform=transform, download=True)
    
    # Split the train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_data_cifar10(batch_size: int) -> tuple:
    # Transform the data by normaizing and transfering to tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32]), transforms.Normalize(0, 1)])

    # Load the data
    train_dataset = torchvision.datasets.CIFAR10(
        root='datasets/cifar10', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root='datasets/cifar10', train=False, transform=transform, download=True)
    
    # Split the train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
