import data_ffcnn as data_ffcnn

from absl import app
from params_ffcnn import FLAGS
from absl import logging

import os
from utils.io import load_params
from utils.perf import timeit

import numpy as np
from sklearn.metrics import accuracy_score


@timeit
def main(argv):
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # load model parameters and features
    llsr_weights = load_params(modelpath, "llsr_weights.pkl")
    llsr_biases = load_params(modelpath, "llsr_biases.pkl")
    feat = load_params(modelpath, "feat.pkl")

    # read data
    _, _, _, test_labels, _ = data_ffcnn.import_data()

    features = feat['testing_feature']
    print("S4 shape:", features.shape)
    print('--------Finish Feature Extraction subnet--------')

    # feature normalization
    std_var = (np.std(features, axis=0)).reshape(1,-1)
    features = features/std_var
    use_dataset = FLAGS.use_dataset
    
    if use_dataset == 'mnist':
        test_mnist(llsr_weights, llsr_biases, features, test_labels)
    elif use_dataset == 'cifar10':
        test_cifar10(llsr_weights, llsr_biases, features, test_labels)


# test network trained on MNIST
def test_mnist(weights, biases, features, test_labels):
    fixed_num_classes = 10  # NOTE use_classes is fixed to use all
    num_clusters = [120, 84, 10]
    for k in range(len(num_clusters)):
        # least square regression
        weight = weights['%d LLSR weight'%k]
        bias = biases['%d LLSR bias'%k]
        features = np.matmul(features, weight)
        features = features+bias
        print(k,' layer LSR weight shape:', weight.shape)
        print(k,' layer LSR bias shape:', bias.shape)
        print(k,' layer LSR output shape:', features.shape)
        
        if k != len(num_clusters)-1:
            pred_labels = np.argmax(features, axis=1)
            num_clas = np.zeros((num_clusters[k], fixed_num_classes))
            for i in range(num_clusters[k]):
                for t in range(fixed_num_classes):
                    for j in range(features.shape[0]):
                        if pred_labels[j] == i and test_labels[j] == t:
                            num_clas[i,t] += 1
            acc_test = np.sum(np.amax(num_clas, axis=1))/features.shape[0]
            print(k,' layer LSR testing acc is {}'.format(acc_test))

            # Relu
            for i in range(features.shape[0]):
                for j in range(features.shape[1]):
                    if features[i,j] < 0:
                        features[i,j] = 0
        else:
            pred_labels = np.argmax(features, axis=1)
            acc_test = accuracy_score(test_labels,pred_labels)
            print('testing acc is {}'.format(acc_test))


# test network trained on CIFAR-10
def test_cifar10(weights, biases, features, test_labels):
    num_clusters = [200, 100, 10]
    for k in range(len(num_clusters)):
        weight = weights['%d LLSR weight'%k]
        bias = biases['%d LLSR bias'%k]
        features = np.matmul(features,weight)+bias
        print(k,' layer LSR weight shape:', weight.shape)
        print(k,' layer LSR output shape:', features.shape)
        if k != len(num_clusters)-1:
            # Relu
            for i in range(features.shape[0]):
                for j in range(features.shape[1]):
                    if features[i,j] < 0:
                        features[i,j] = 0
        else:
            pred_labels = np.argmax(features, axis=1)
            acc_test = accuracy_score(test_labels, pred_labels)
            print('testing acc is {}'.format(acc_test))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass