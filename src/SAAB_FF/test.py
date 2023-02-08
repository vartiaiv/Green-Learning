import data

import pickle
import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances

from utils.timer import timeit

@timeit
def main():
    with open(r'./model/llsr_weights.pkl','rb') as fr:
        weights = pickle.load(fr, encoding='latin1')
    with open(r'./model/llsr_bias.pkl','rb') as fr:
        biases = pickle.load(fr, encoding='latin1')
    # read data
    _, _, _, test_labels, _ = data.import_data("0-9")
    
    # load feature
    with open(r'./model/feat.pkl','rb') as fr:
        feat = pickle.load(fr, encoding='latin1')
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
    main()