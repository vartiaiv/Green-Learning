import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import numpy as np
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10, MNIST
from sklearn.model_selection import train_test_split

import saab


def import_data():
    """ Return training and testing images, labels and a classlist """
    # choose corresponding torchvision.datasets function
    use_dataset = FLAGS.use_dataset
    datasets_dict = {'cifar10': CIFAR10, 'mnist': MNIST}  
    data_root = os.path.join(FLAGS.datasets_root, use_dataset)
    print(f"Importing data from: {data_root}")

    # selected classes
    use_classes = FLAGS.use_classes
    class_list = [0,1,2,3,4,5,6,7,8,9]  # default
    if use_classes != "0-9":
        class_list = saab.parse_list_string(use_classes)


    DATASET = datasets_dict[use_dataset]
    T = Compose([
        # ToTensor()
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
    train_images = preprocess_images(train_images, (32, 32))
    test_images = preprocess_images(test_images, (32, 32))

    if 0 < FLAGS.use_portion < 1.0:
        use_portion = FLAGS.use_portion
        num_train_images = round(use_portion * len(train_set))  # subset size
        num_test_images = round(use_portion * len(test_set))  # subset size
        # select balanced subsets (classes uniform distributed)
        seed = 0  # for reproducible subset selection!
        train_images, train_labels = saab.select_balanced_subset(train_images, train_labels, num_train_images, class_list, shuffle_seed=seed)
        test_images, test_labels = saab.select_balanced_subset(train_images, train_labels, num_test_images, class_list, shuffle_seed=seed)
    else:
        raise("bad portion: needs to be strictly between 0 and 1.0")

    train_images, train_labels = get_data_for_class(train_images, train_labels, class_list)
    test_images, test_labels = get_data_for_class(test_images, test_labels, class_list)
    
    print('Training image size:', train_images.shape)  
    print('Testing_image size:', test_images.shape)

    return train_images, train_labels, test_images, test_labels, class_list    


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
    if diff_h >= 0 and diff_w >= 0:
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
    import_data()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass