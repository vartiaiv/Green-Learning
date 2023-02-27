# import torch
from torch.utils import data
from torchvision import transforms as tf
from torchvision import datasets

from matplotlib import pyplot as plt

# for testing FF_CNN perf utilities
from src.FF_CNN.utils.perf import timeit  # decorator for timing functions


@timeit
def LoadData(num_batch=64):
    ''' Load datasets into data_root and transform them into PyTorch tensors.
    Dataset is loaded only if it does not exists in data_root.
    '''
    data_root = r'./datasets'
    T = tf.Compose([

        # TODO Normalizations?

        tf.ToTensor()
    ])
    train_data = datasets.MNIST(data_root, train=True, download=True, transform=T, target_transform=None)
    val_data = datasets.MNIST(data_root, train=False, download=True, transform=T, target_transform=None)
    train_dl = data.DataLoader(train_data, batch_size=num_batch)
    valid_dl = data.DataLoader(val_data, batch_size=num_batch)    
    return train_data, val_data, train_dl, valid_dl

train_data, val_data, train_dl, valid_dl = LoadData()
# plt.imshow(train_data[0][0][0], cmap='gray')
