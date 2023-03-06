import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import numpy as np
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST

import saab


def import_data(train=True):
    """ Return images, labels and a classlist """
    
    # choose corresponding torchvision.datasets function
    datasets_dict = {'cifar10': CIFAR10, 'mnist': MNIST}
    DATASET = datasets_dict[FLAGS.use_dataset]
    T = Compose([
        # ToTensor()
    ])
    data_root = os.path.join(FLAGS.datasets_root, FLAGS.use_dataset)
    print(f"Importing data from: {data_root}")
    data_set = DATASET(data_root, train=train, download=True, transform=T, target_transform=None)
    
    # selected classes
    class_list = [0,1,2,3,4,5,6,7,8,9]  # default all 10 classes

    # extract images as tensors
    images = data_set.data    # train_images = train_set.data

    # extract labels as numpy arrays
    labels = np.array(data_set.targets)

    # zero pad images into 32 by 32
    # also add missing 4th dimension for non-RGB images
    # cifar10 images are already this size so the function doesn't do anything
    images = preprocess_images(images, (32, 32))

    images, labels = get_data_for_class(images, labels, class_list)
    # print(f"{'Training' if train else 'Testing'} image size:", images.shape)  

    return images, labels, class_list


def preprocess_images(images, to_size=(32, 32)):
    """ Preprocessing for images
    - Adds missing array dimension for "1d images".
    - Normalize image values by dividing with 255
    - Zero padding into given size if needed.
    """
    # Add the missing dimension if image has only luminocity channel such as MNIST
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=3)
           
    _, h, w, _ = images.shape

    # normalize image values between 0 and 1
    images = images / 255.

    # Calculate the needed padding
    diff_h = (to_size[0]-h)
    diff_w = (to_size[1]-w)
    if diff_h <= 0 and diff_w <= 0:
        return images  # don't do anything

    # Calculate axis-wise padding and add any surplus to the bottom-right corner
    pad_h1 = diff_h // 2
    pad_h2 = diff_h // 2 + diff_h % 2
    pad_h = (pad_h1, pad_h2)
    
    pad_w1 = diff_w // 2
    pad_w2 = diff_w // 2 + diff_w % 2  
    pad_w = (pad_w1, pad_w2)
        
    padded_images = np.pad(images, ((0,0), pad_h, pad_w, (0,0)), mode='constant')

    return padded_images


def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]


# debug
def main(argv):
    import_data(train=True)
    import_data(train=False)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass