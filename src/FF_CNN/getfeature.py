import data
import saab

from absl import app
from absl import logging
from defaultflags import FLAGS

import matplotlib.pyplot as plt

from utils.perf import timeit

import os
from src.utils.io import save_params, load_params

here = os.path.dirname(os.path.abspath(__file__))

@timeit
def main(argv):
    modelpath = os.path.join(here, f"{FLAGS.use_dataset}_model")

    # load pca params obtained from getkernel
    pca_params = load_params(modelpath, "pca_params.pkl")

    # read data
    train_images, _, test_images, _, _ = data.import_data()

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