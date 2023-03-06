import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import saab
import data_ffcnn
from utils.io import save_params, load_params
from utils.perf import mytimer

import numpy as np

@mytimer
def main(argv):    
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # read data
    train_images, train_labels, _ = data_ffcnn.import_data()
    test_images, test_labels, _ = data_ffcnn.import_data(train=False)

    # get features with PCA
    pca_params = load_params(modelpath, "pca_params.pkl")

    print('--------Extract Training features--------')
    train_feat = saab.initialize(train_images, pca_params)
    train_feat = train_feat.reshape(train_feat.shape[0], -1)
    # feature normalization
    std_var = (np.std(train_feat, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    train_feat = train_feat/std_var
    print("S4 training features shape:", train_feat.shape)
    print('--------Finish Feature Extraction subnet--------')

    print('--------Extract Testing features--------')
    test_feat = saab.initialize(test_images, pca_params)
    test_feat = test_feat.reshape(test_feat.shape[0], -1)
    # feature normalization
    std_var = (np.std(test_feat, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    test_feat = test_feat/std_var
    print("S4 test features shape:", test_feat.shape)
    print('--------Finish Feature Extraction subnet--------')

    save_params(modelpath, "train_feat.pkl", train_feat)
    save_params(modelpath, "train_labels.pkl", train_labels)
    save_params(modelpath, "test_feat.pkl", test_feat)
    save_params(modelpath, "test_labels.pkl", test_labels)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass