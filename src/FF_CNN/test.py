import data_ffcnn as data_ffcnn
import saab

from absl import app
from params_ffcnn import FLAGS
from absl import logging

import os
from utils.io import load_params
from utils.perf import mytimer
import timeit

import numpy as np
from sklearn.metrics import accuracy_score


@mytimer
def main(argv):
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # read data
    _, _, test_images, test_labels, class_list = data_ffcnn.import_data()

    # get features with PCA
    pca_params = load_params(modelpath, "pca_params.pkl")
    test_feat = saab.initialize(test_images, pca_params)
    test_feat = test_feat.reshape(test_feat.shape[0], -1)
    print("S4 shape:", test_feat.shape)
    print('--------Finish Feature Extraction subnet--------')

    # feature normalization
    std_var = (np.std(test_feat, axis=0)).reshape(1,-1)
    test_feat = test_feat/std_var
    num_classes = len(class_list)
    num_clusters = saab.parse_list_string(FLAGS.num_clusters)

    # load model parameters
    llsr_weights = load_params(modelpath, "llsr_weights.pkl")
    llsr_biases = load_params(modelpath, "llsr_biases.pkl")

    # Start testing
    for k in range(len(num_clusters)):
        weight = llsr_weights['%d LLSR weight'%k]
        bias = llsr_biases['%d LLSR bias'%k]
        test_feat = np.matmul(test_feat,weight)+bias
        print(k,' layer LSR weight shape:', weight.shape)
        print(k,' layer LSR bias shape:', bias.shape)
        print(k,' layer LSR output shape:', test_feat.shape)

        # --------------- When using MNIST -------------------------------
        if FLAGS.use_dataset == 'mnist':
            pred_labels=np.argmax(test_feat, axis=1)
            num_clas=np.zeros((num_clusters[k], num_classes))
            for i in range(num_clusters[k]):
                for t in range(num_classes):
                    for j in range(test_feat.shape[0]):
                        if pred_labels[j]==i and test_labels[j]==t:
                            num_clas[i,t]+=1
            acc_test=np.sum(np.amax(num_clas, axis=1))/test_feat.shape[0]
            print(k,' layer LSR testing acc is {}'.format(acc_test))
        # ----------------------------------------------------------------

        if k != len(num_clusters)-1:
            # Relu
            for i in range(test_feat.shape[0]):
                for j in range(test_feat.shape[1]):
                    if test_feat[i,j] < 0:
                        test_feat[i,j] = 0
        else:
            pred_labels = np.argmax(test_feat, axis=1)
            acc_test = accuracy_score(test_labels, pred_labels)
            print('testing acc is {}'.format(acc_test))        


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass