import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import saab
import data_ffcnn as data_ffcnn
from utils.io import save_params, load_params
from utils.perf import timeit


@timeit
def main(argv):
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # load pca params obtained from getkernel
    pca_params = load_params(modelpath, "pca_params.pkl")

    # read data
    train_images, _, test_images, _, _ = data_ffcnn.import_data()

    feat = {}
    # Features for training
    print('--------Training--------')
    train_feat = saab.initialize(train_images, pca_params) 
    print("S4 shape:", train_feat.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['training_feature'] = train_feat

    # Features for testing
    print('--------Testing--------')
    test_feat = saab.initialize(test_images, pca_params) 
    test_feat=test_feat.reshape(test_feat.shape[0],-1)
    print("S4 shape:", test_feat.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['testing_feature'] = test_feat

    save_params(modelpath, "feat.pkl", feat)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass