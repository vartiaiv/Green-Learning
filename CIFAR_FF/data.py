import saab

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
from utils.padder import pad_to_size
from utils.timer import timeit

dataset_func = {'cifar10': CIFAR10, 'mnist': MNIST}


# def pad_to_size(images, to_size=(32, 32)):
#     """ A general zero padding function into given size if needed.
#     - Also adds a missing dimension for black and white images.
#     - Assumes that the images are of size <= 32

#     - If this function is overkill just replace  the calls with the following
#     if use_dataset == 'mnist':
#         train_images = np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
#         test_images = np.pad(test_images,  ((0,0),(2,2),(2,2),(0,0)), mode='constant')
#     """
#     # Add the missing dimension if image has only luminocity channel such as MNIST
#     if len(images.shape) == 3:
#         images = np.expand_dims(images, axis=3)
           
#     _, h, w, _ = images.shape

#     # Calculate the needed padding
#     diff_h = (to_size[0]-h)
#     diff_w = (to_size[1]-w)
#     if diff_h == 0 and diff_w == 0:
#         return images  # don't do anything

#     # Calculate axis-wise padding and add any surplus to the bottom-right corner
#     pad_h1 = diff_h // 2
#     pad_h2 = diff_h // 2 + diff_h % 2
#     pad_h = (pad_h1, pad_h2)
    
#     pad_w1 = diff_w // 2
#     pad_w2 = diff_w // 2 + diff_w % 2  
#     pad_w = (pad_w1, pad_w2)
        
#     padded_images = np.pad(images, ((0,0), pad_h, pad_w, (0,0)), mode='constant')

#     return padded_images

def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]

def import_data(use_classes, use_dataset):
    # data_root = r'../datasets'
    data_root = r'./datasets'
    if not os.path.exists(data_root):
        raise Exception("Data root directory missing")

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