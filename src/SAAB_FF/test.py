import data

from absl import app
from absl import logging
from defaultflags import FLAGS

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from src.utils.timer import timeit

import os
from src.utils.io import save, load

# io paths
here = os.path.dirname(os.path.abspath(__file__))
loadpath_weights = os.path.join(here, "model", "llsr_weights.pkl")
loadpath_bias = os.path.join(here, "model", "llsr_bias.pkl")
loadpath_feat = os.path.join(here, "model", "feat.pkl")


@timeit
def main(argv):
    # load model parameters and features
    weights = load(loadpath_weights)
    biases = load(loadpath_bias)
    feat = load(loadpath_feat)

    # read data
    use_classes = FLAGS.use_classes
    use_dataset = FLAGS.use_dataset
    use_portion = FLAGS.use_portion
    _, _, _, test_labels, _ = data.import_data(use_classes, use_dataset, use_portion)

    feature = feat['testing_feature']
    feature = feature.reshape(feature.shape[0],-1)
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')

    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1,-1)
    feature = feature/std_var

    num_clusters = [200, 100, 10]
    # use_classes = 10
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
            acc_test = sklearn.metrics.accuracy_score(test_labels, pred_labels)
            print('testing acc is {}'.format(acc_test))

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass