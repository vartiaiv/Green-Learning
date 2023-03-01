import os
# NOTE Comment the next line and see if any warnings come from KMeans
# If needed, do this BEFORE importing numpy and KMeans or any other module using them
# Avoid memory leak with KMeans on Windows with MKL, when less chunks than available threads
os.environ["OMP_NUM_THREADS"] = '1'  

from absl import app
from params_ffcnn import FLAGS
from absl import logging
import saab

import data_ffcnn
from utils.io import save_params, load_params
from utils.perf import timeit

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from sklearn.metrics import accuracy_score
from numpy import linalg as LA


def to_categorical(y, num_classes):
    # 1-hot encode a tensor similar to keras.utils.to_categorical"""
    return np.eye(num_classes, dtype='uint8')[y]


@timeit
def main(argv):
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # load features
    feat = load_params(modelpath, "feat.pkl")

    # read data
    _, train_labels, _, _, class_list = data_ffcnn.import_data()

    train_feat = feat['training_feature']
    
    train_feat = train_feat.reshape(train_feat.shape[0], -1)
    print("S4 shape:", train_feat.shape)
    print('--------Finish Feature Extraction subnet--------')
    
    # feature normalization
    std_var = (np.std(train_feat, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    train_feat = train_feat/std_var  

    num_clusters = saab.parse_list_string(FLAGS.num_clusters)
    num_classes = len(class_list)  # expect to use all classes
    
    llsr_weights = {}
    llsr_biases = {}
    if FLAGS.use_dataset == 'mnist':
        helper = mnist_helper
    elif FLAGS.use_dataset == 'cifar10':
        helper = cifar10_helper

    # NOTE Testing starts here        
    for k in range(len(num_clusters)):
        if k != len(num_clusters)-1:
            # this is only difference between mnist and cifar10 versions
            labels, pred_labels = \
                helper(k, train_feat, train_labels, num_classes, num_clusters)  

            # least square regression
            A = np.ones((train_feat.shape[0],1))
            train_feat = np.concatenate((A,train_feat), axis=1)
            weight = np.matmul(LA.pinv(train_feat), labels)
            train_feat = np.matmul(train_feat,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_biases['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', train_feat.shape)

            pred_labels = np.argmax(train_feat, axis=1)
            num_clas = np.zeros((num_clusters[k], num_classes))
            for i in range(num_clusters[k]):
                for t in range(num_classes):
                    for j in range(train_feat.shape[0]):
                        if pred_labels[j] == i and train_labels[j] == t:
                            num_clas[i,t] += 1
            
            # extra print
            acc_train = np.sum(np.amax(num_clas, axis=1))/train_feat.shape[0]
            print(k,' layer LSR training acc is {}'.format(acc_train))

            # Relu
            for i in range(train_feat.shape[0]):
                for j in range(train_feat.shape[1]):
                    if train_feat[i,j]<0:
                        train_feat[i,j]=0

        else:  # last round
            # least square regression
            labels = to_categorical(train_labels, num_classes)
            A = np.ones((train_feat.shape[0],1))
            train_feat = np.concatenate((A,train_feat),axis=1)
            weight = np.matmul(LA.pinv(train_feat),labels)
            train_feat = np.matmul(train_feat,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_biases['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', train_feat.shape)
            
            # predictions
            pred_labels = np.argmax(train_feat, axis=1)
            acc_train = accuracy_score(train_labels, pred_labels)
            print('training acc is {}'.format(acc_train))

    save_params(modelpath, "llsr_weights.pkl", llsr_weights)
    save_params(modelpath, "llsr_biases.pkl", llsr_biases)



def mnist_helper(k, train_feat, train_labels, num_classes, num_clusters):
    # Kmeans_Mixed_Class
    kmeans = KMeans(n_clusters = num_clusters[k]).fit(train_feat)
    pred_labels = kmeans.labels_
    num_class_clusters = np.zeros((num_clusters[k], num_classes))
    for i in range(num_clusters[k]):
        for t in range(num_classes):
            for j in range(train_feat.shape[0]):
                if pred_labels[j] == i and train_labels[j] == t:
                    num_class_clusters[i,t] += 1
    
    # NOTE extra print, not needed
    # acc_train = np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
    # print(k,' layer Kmean (just ref) training acc is {}'.format(acc_train))

    # Compute centroids
    clus_labels = np.argmax(num_class_clusters, axis=1)
    centroid = np.zeros((num_clusters[k], train_feat.shape[1]))
    for i in range(num_clusters[k]):
        t = 0
        for j in range(train_feat.shape[0]):
            if pred_labels[j] == i and clus_labels[i] == train_labels[j]:
                if t == 0:
                    feat_tmp = train_feat[j].reshape(1,-1)
                else:
                    feat_tmp = np.concatenate((feat_tmp, train_feat[j].reshape(1,-1)), axis=0)
                t += 1
        centroid[i] = np.mean(feat_tmp, axis=0, keepdims=True)

    # Compute one hot vector
    labels = np.zeros((train_feat.shape[0], num_clusters[k]))  # LABELS
    for i in range(train_feat.shape[0]):
        if clus_labels[pred_labels[i]] == train_labels[i]:
            labels[i,pred_labels[i]] = 1
        else:
            # distance_assigned = euclidean_distances(feature[i].reshape(1,-1), centroid[pred_labels[i]].reshape(1,-1))
            cluster_special = [j for j in range(num_clusters[k]) if clus_labels[j] == train_labels[i]]
            distance = np.zeros(len(cluster_special))
            for j in range(len(cluster_special)):
                distance[j] = euclidean_distances(train_feat[i].reshape(1,-1), centroid[cluster_special[j]].reshape(1,-1))
            labels[i, cluster_special[np.argmin(distance)]] = 1
    
    return labels, pred_labels


def cifar10_helper(k, feature, train_labels, num_classes, num_clusters):
    # (Mixed class KMeans too slow for CIFAR, changed into Fixed Class)
    num_class_clusters = int(num_clusters[k]/num_classes)
    labels = np.zeros((feature.shape[0], num_clusters[k]))
    
    for cid in range(num_classes):
        idx = (train_labels == cid)
        index=np.where(idx == True)[0]
        feature_special=np.zeros((index.shape[0],feature.shape[1]))
        for i in range(index.shape[0]):
            feature_special[i] = feature[index[i]]
        kmeans = KMeans(n_clusters=num_class_clusters).fit(feature_special)  # error?
        pred_labels = kmeans.labels_
        for i in range(feature_special.shape[0]):
            labels[index[i], pred_labels[i] + cid*num_class_clusters] = 1

    return labels, pred_labels

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass