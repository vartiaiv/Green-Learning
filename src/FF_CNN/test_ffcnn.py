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
    weights = load_params(modelpath, "llsr_weights.pkl")
    biases = load_params(modelpath, "llsr_biases.pkl")
    feat = load_params(modelpath, "feat.pkl")

    # read data
    _, _, _, test_labels, _ = data_ffcnn.import_data()

    feature = feat['testing_feature']
    feature = feature.reshape(feature.shape[0],-1)
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')

    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1,-1)
    feature = feature/std_var

    num_clusters = [200, 100, 10]
    for k in range(len(num_clusters)):
        weight = weights['%d LLSR weight'%k]
        bias = biases['%d LLSR bias'%k]
        feature = np.matmul(feature,weight)+bias
        print(k,' layer LSR weight shape:', weight.shape)
        print(k,' layer LSR output shape:', feature.shape)
        if k != len(num_clusters)-1:
            # Relu
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i,j] < 0:
                        feature[i,j] = 0
        else:
            pred_labels = np.argmax(feature, axis=1)
            acc_test = accuracy_score(test_labels, pred_labels)
            print('testing acc is {}'.format(acc_test))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass