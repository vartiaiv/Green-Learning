import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

# NOTE Avoid memory leak with KMeans on windows with MKL, when less chunks than available threads
# NOTE do this BEFORE importing numpy and KMeans or any other module using them
os.environ["OMP_NUM_THREADS"] = '2'  

import data_ffcnn
from utils.io import save_params, load_params
from utils.perf import timeit

import numpy as np
from sklearn.cluster import KMeans
import sklearn
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

    feature = feat['training_feature']
    
    feature = feature.reshape(feature.shape[0], -1)
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    
    # feature normalization
    std_var = (np.std(feature, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    feature = feature/std_var  

    num_clusters = [200, 100, 10]
    llsr_weights = {}
    llsr_biases = {}
    for k in range(len(num_clusters)):
        if k != len(num_clusters)-1:
            num_clus = int(num_clusters[k]/len(class_list))
            labels = np.zeros((feature.shape[0], num_clusters[k]))
            
            for n in range(len(class_list)):
                idx=(train_labels==class_list[n])
                index=np.where(idx==True)[0]
                feature_special=np.zeros((index.shape[0],feature.shape[1]))
                for i in range(index.shape[0]):
                    feature_special[i]=feature[index[i]]
                kmeans=KMeans(n_clusters=num_clus).fit(feature_special)  # error
                pred_labels=kmeans.labels_
                for i in range(feature_special.shape[0]):
                    labels[index[i],pred_labels[i]+n*num_clus]=1

            # least square regression
            A = np.ones((feature.shape[0],1))
            feature = np.concatenate((A,feature),axis=1)
            weight = np.matmul(LA.pinv(feature),labels)
            feature = np.matmul(feature,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_biases['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            num_clas = np.zeros((num_clusters[k],len(class_list)))
            for j in range(num_clusters[k]):
                for t in range(len(class_list)):
                    for j in range(feature.shape[0]):
                        if pred_labels[j] == j and train_labels[j] == t:
                            num_clas[j,t] += 1
            acc_train = np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
            print(k,' layer LSR training acc is {}'.format(acc_train))

            # Relu
            for j in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[j,j] < 0:
                        feature[j,j] = 0
        else:
            # linear least square regression (llsr)
            labels = to_categorical(train_labels, 10)
            A = np.ones((feature.shape[0],1))
            feature = np.concatenate((A,feature),axis=1)
            weight = np.matmul(LA.pinv(feature),labels)
            feature = np.matmul(feature,weight)
            llsr_weights['%d LLSR weight'%k] = weight[1:weight.shape[0]]
            llsr_biases['%d LLSR bias'%k] = weight[0].reshape(1,-1)
            print(k,' layer LSR weight shape:', weight.shape)
            print(k,' layer LSR output shape:', feature.shape)
            
            pred_labels = np.argmax(feature, axis=1)
            acc_train = sklearn.metrics.accuracy_score(train_labels,pred_labels)
            print('training acc is {}'.format(acc_train))

    save_params(modelpath, "llsr_weights.pkl", llsr_weights)
    save_params(modelpath, "llsr_biases.pkl", llsr_biases)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass