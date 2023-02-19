import data

from absl import app
from absl import logging
from defaultflags import FLAGS

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances

from utils.timer import timeit

import os
from src.utils.io import save, load

# io paths
here = os.path.dirname(os.path.abspath(__file__))
loadpath = os.path.join(here, "model", "feat.pkl")
savepath_weights = os.path.join(here, "model", "llsr_weights.pkl")
savepath_bias = os.path.join(here, "model", "llsr_bias.pkl")

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor. Alternative to keras.utils.to_categorical"""
    return np.eye(num_classes, dtype='uint8')[y]

@timeit
def main(argv):
    # load features
    feat = load(loadpath)

    # read data
    _, train_labels, _, _, class_list = data.import_data("0-9")

    feature = feat['training_feature']
    
    feature = feature.reshape(feature.shape[0], -1)
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    
    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1,-1)
    feature = feature/std_var

    num_clusters = [200, 100, 10]
    use_classes = class_list
    llsr_weights = {}
    llsr_bias = {}
    for k in range(len(num_clusters)):
        if k != len(num_clusters)-1:
            num_clus = int(num_clusters[k]/len(use_classes))
            labels = np.zeros((feature.shape[0], num_clusters[k]))
            
            for n in range(len(use_classes)):
                idx = (train_labels == use_classes[n])
                index = np.where(idx == True)[0]
                feature_special = np.zeros((index.shape[0],feature.shape[1]))

                for i in range(index.shape[0]):
                    feature_special[i] = feature[index[i]]
                kmeans = KMeans(n_clusters=num_clus).fit(feature_special.reshape(-1,1))  # edited
                pred_labels = kmeans.labels_
                
                for i in range(feature_special.shape[0]):
                    labels[index[i],pred_labels[i]+n*num_clus]=1

            # least square regression
            A = np.ones((feature.shape[0],1))
            feature = np.concatenate((A,feature),axis=1)
            weight = np.matmul(LA.pinv(feature),labels)
            feature = np.matmul(feature,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_bias['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            num_clas = np.zeros((num_clusters[k],len(use_classes)))
            for i in range(num_clusters[k]):
                for t in range(len(use_classes)):
                    for j in range(feature.shape[0]):
                        if pred_labels[j] == i and train_labels[j] == t:
                            num_clas[i,t] += 1
            acc_train = np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
            print(k,' layer LSR training acc is {}'.format(acc_train))

            # Relu
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i,j] < 0:
                        feature[i,j] = 0
        else:
            # linear least square regression (llsr)
            labels = to_categorical(train_labels,10)
            A = np.ones((feature.shape[0],1))
            feature = np.concatenate((A,feature),axis=1)
            weight = np.matmul(LA.pinv(feature),labels)
            feature = np.matmul(feature,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_bias['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', feature.shape)
            
            pred_labels = np.argmax(feature, axis=1)
            acc_train = sklearn.metrics.accuracy_score(train_labels,pred_labels)
            print('training acc is {}'.format(acc_train))
    
    # save data
    save(savepath_weights, llsr_weights)
    save(savepath_bias, llsr_bias)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass