import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import saab
import data_ffcnn as data_ffcnn
from params_ffcnn import MODELS_ROOT
from utils.io import save_params, load_params
from utils.perf import timeit, mem_profile


@timeit
@mem_profile
def main(argv):
    # io paths
    modelpath = os.path.join(MODELS_ROOT, f"ffcnn_{FLAGS.use_dataset}")

    # load pca params obtained from getkernel
    pca_params = load_params(modelpath, "pca_params.pkl")

    # read data
    train_images, _, test_images, _, _ = data_ffcnn.import_data()

    feat = {}
    # Features for training
    print('--------Training--------')
    feature = saab.initialize(train_images, pca_params) 
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['training_feature']=feature

    # Features for testing
    print('--------Testing--------')
    feature = saab.initialize(test_images, pca_params) 
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['testing_feature'] = feature

    save_params(modelpath, "feat.pkl", feat)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass