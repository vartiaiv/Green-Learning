# import keras
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10
import numpy as np
import saab

def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]

def import_data(use_classes, num_batch=64):
    data_root = r'./datasets'
    T = Compose([
        # TODO Normalizations?
        ToTensor()
    ])
    (train_images, train_labels) = CIFAR10(data_root, train=True, download=True, transform=T, target_transform=None)
    (test_data, test_labels) = CIFAR10(data_root, train=False, download=True, transform=T, target_transform=None)
    train_loader = DataLoader(train_images, batch_size=num_batch)
    test_loader = DataLoader(test_data, batch_size=num_batch)  

    train_images = train_images/255.
    test_images = test_images/255.
    # print(train_images.shape) # 50000*32*32*3

    if use_classes != '0-9':
        class_list = saab.parse_list_string(use_classes)
        train_images, train_labels = get_data_for_class(train_images, train_labels, class_list)
        test_images, test_labels = get_data_for_class(test_images, test_labels, class_list)
        # print(class_list)
    else:
        class_list = [0,1,2,3,4,5,6,7,8,9]
        
    return train_images, train_labels, test_images, test_labels, class_list