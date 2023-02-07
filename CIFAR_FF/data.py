import saab

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10
import numpy as np
import torch

def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]

def import_data(use_classes, num_batch=64):
    data_root = r'../datasets'
    T = Compose([
        # TODO Normalizations?
        ToTensor()
    ])
    train_set = CIFAR10(data_root, train=True, download=True, transform=T, target_transform=None)
    test_set = CIFAR10(data_root, train=False, download=True, transform=T, target_transform=None)
    train_loader = DataLoader(train_set, batch_size=num_batch)
    test_loader = DataLoader(test_set, batch_size=num_batch)  

    train_images = next(iter(train_loader))[0]
    test_images = next(iter(test_loader))[0]
    train_labels = next(iter(train_loader))[1]
    test_labels = next(iter(test_loader))[1]
    train_images = torch.mul(train_images, 1/255).permute(0, 2, 3, 1) # train_images / 255.
    test_images = torch.mul(test_images, 1/255).permute(0, 2, 3, 1)  # test_images / 255. 
    print(train_images.shape) # 50000*3*32*32  # default pytorch tensor
    

    # train_images = train_set.data[0:num_batch]
    # test_images = test_set.data[0:num_batch]
    # train_labels = train_set.targets[0:num_batch]
    # test_labels = test_set.targets[0:num_batch]
    # train_images /= 255.
    # test_images /= 255.
    # print(train_images.shape) # 50000*32*32*3  # default numpy ndarray

    if use_classes != "0-9":
        class_list = saab.parse_list_string(use_classes)
        train_images, train_labels = get_data_for_class(train_images, train_labels, class_list)
        test_images, test_labels = get_data_for_class(test_images, test_labels, class_list)
        # print(class_list)
    else:
        class_list = [0,1,2,3,4,5,6,7,8,9]
        
    return train_images, train_labels, test_images, test_labels, class_list