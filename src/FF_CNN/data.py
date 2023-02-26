""" Create new data cache when run.
Call import_data when cache doesn't need to be updated!
"""
import saab

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST
import numpy as np

from absl import app
from absl import logging
from defaultflags import FLAGS

import os
from src.utils.padder import pad_to_size
from src.utils.io import join_path_common


here = os.path.dirname(os.path.abspath(__file__))  # this directory
data_rel_path = "Green Learning/datasets"  # data location w.r.t. top level
data_root = join_path_common(here, data_rel_path)  # path to 'datasets'


def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]


def import_data():
    """ Return training and testing images, labels and a classlist """
    
    use_dataset = FLAGS.use_dataset
    use_classes = FLAGS.use_classes
    use_portion = FLAGS.use_portion

    print(f"Importing data from: {data_root}")
    # it's good practice to create the datasets directory before downloading
    assert(os.path.exists(data_root))  # makes sure that the datasets directory exists

    class_list = [0,1,2,3,4,5,6,7,8,9]  # default
    if use_classes != "0-9":
        class_list = saab.parse_list_string(use_classes)

    # choose corresponding torchvision.datasets function
    dataset_dict = {'cifar10': CIFAR10, 'mnist': MNIST}  
    DATASET = dataset_dict[use_dataset]
    T = Compose([
        # TODO Normalizations?
        ToTensor()
    ])
    train_set = DATASET(data_root, train=True, download=True, transform=T, target_transform=None)
    test_set = DATASET(data_root, train=False, download=True, transform=T, target_transform=None)
    
    # extract images as tensors
    train_images = train_set.data    # train_images = train_set.data
    test_images = test_set.data      # test_images = test_set.data

    # extract labels as numpy arrays
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # zero pad images into 32 by 32
    # also add missing 4th dimension for non-RGB images
    # cifar10 images are already this size so the function doesn't do anything
    train_images = pad_to_size(train_images, (32, 32))
    test_images = pad_to_size(test_images, (32, 32))

    # select balanced subsets (expects 4-dimensional image data)
    # i.e. the class distribution is uniform!
    num_train_images = round(use_portion * len(train_images))
    num_test_images = round(use_portion * len(test_images))
    train_images, train_labels = saab.select_balanced_subset(train_images, train_labels, num_train_images, class_list)
    test_images, test_labels = saab.select_balanced_subset(train_images, train_labels, num_test_images, class_list)

    # NOTE for debug
    # counts = [sum(1 for k in selected_labels[selected_labels == c]) for c in class_list]
    # print(f"Class distribution: {counts}")
    # print(selected_images.shape)
    # print(selected_labels[0:10])

    # scale image values
    train_images = train_images / 255.
    test_images = test_images / 255.

    print('Training image size:', train_images.shape)  
    print('Testing_image size:', test_images.shape)
    # cifar10 50000*32*32*3, 
    # mnist 60000*28*28*1

    train_images, train_labels = get_data_for_class(train_images, train_labels, class_list)
    test_images, test_labels = get_data_for_class(test_images, test_labels, class_list)

    return train_images, train_labels, test_images, test_labels, class_list    


def main(argv):
    import_data()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass