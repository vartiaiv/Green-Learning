import saab

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
from utils.padder import pad_to_size
from utils.io import mkdir_new
# from utils.timer import timeit

dataset_func = {'cifar10': CIFAR10, 'mnist': MNIST}

def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]

def import_data(use_classes, use_dataset):
    # data root assumes working directory to be src
    print(__file__)
    cwd = os.getcwd().split("\\")[-1]
    if cwd != "src":
        raise Exception("Working directory should be src")
    data_root = r'../datasets'
    mkdir_new(data_root)
    T = Compose([
        # TODO Normalizations?
        ToTensor()
    ])
    DATASET = dataset_func[use_dataset]
    train_set = DATASET(data_root, train=True, download=True, transform=T, target_transform=None)
    test_set = DATASET(data_root, train=False, download=True, transform=T, target_transform=None)
    
    # extract images
    test_images = test_set.data
    train_images = train_set.data

    # zero pad images into 32 by 32
    # cifar10 images are already this size so the function doesn't do anything
    train_images = pad_to_size(train_images, (32, 32))
    test_images = pad_to_size(test_images, (32, 32))

    # extract labels
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # scale image values
    train_images = train_images / 255.
    test_images = test_images / 255.

    print('Training image size:', train_images.shape)  
    print('Testing_image size:', test_images.shape)
    # cifar10 50000*32*32*3, 
    # mnist 60000*28*28*1

    if use_classes != "0-9":
        class_list = saab.parse_list_string(use_classes)
        train_images, train_labels = get_data_for_class(train_images, train_labels, class_list)
        test_images, test_labels = get_data_for_class(test_images, test_labels, class_list)
        # print(class_list)
    else:
        class_list = [0,1,2,3,4,5,6,7,8,9]
        
    return train_images, train_labels, test_images, test_labels, class_list


# Debug only: hardcoded input for testing import_data(), nothing else
def main():
    for dataset_name in ['mnist', 'cifar10']:
        print(f"testing import_data('0-9', '{dataset_name}')")
        import_data('0-9', dataset_name)

if __name__ == "__main__":
    main()