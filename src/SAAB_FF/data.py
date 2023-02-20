import saab

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST
import numpy as np

from src.utils.padder import pad_to_size

import os
from src.utils.io import mkdir_new, join_from_common

# io paths
here = os.path.dirname(os.path.abspath(__file__))
data_rel_path = "Green Learning/datasets"
data_root = join_from_common(here, data_rel_path)

def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]


def import_data(use_classes, use_dataset, use_portion
    ):
    """ Return training and testing images, labels and a classlist """
    T = Compose([
        # TODO Normalizations?
        ToTensor()
    ])
    print(f"Importing data from: {data_root}")
    # it's good practice to create the datasets directory before downloading
    assert(os.path.exists(data_root))  # makes sure that the datasets directory exists
    # mkdir_new(data_root)

    class_list = [0,1,2,3,4,5,6,7,8,9]  # default
    if use_classes != "0-9":
        class_list = saab.parse_list_string(use_classes)
    # print(class_list)
    
    # choose corresponding torchvision.datasets function
    dataset_func = {'cifar10': CIFAR10, 'mnist': MNIST}
    DATASET = dataset_func[use_dataset]  
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


# NOTE debug only: hardcoded input for testing import_data(), nothing else
def main():
    for dataset_name in ['mnist', 'cifar10']:
        print(f"testing import_data('0-9', '{dataset_name}')")
        import_data('0-9', dataset_name, 0.1)

if __name__ == "__main__":
    main()