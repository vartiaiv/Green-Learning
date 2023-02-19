import data
import saab

from absl import app
from absl import logging
from defaultflags import FLAGS

import matplotlib.pyplot as plt

from src.utils.timer import timeit

import os
from src.utils.io import save, load

# io paths
here = os.path.dirname(os.path.abspath(__file__))
loadpath = os.path.join(here, "model", "pca_params.pkl")
savepath = os.path.join(here, "model", "feat.pkl")

@timeit
def main(argv):
    # load pca params from getkernel
    pca_params = load(loadpath)    

    # read data
    train_images, _, test_images, _, _ = data.import_data("0-9")

    feat = {}
    # Training
    print('--------Training--------')
    feature = saab.initialize(train_images, pca_params) 
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['training_feature']=feature

    print('--------Testing--------')
    feature = saab.initialize(test_images, pca_params) 
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')
    feat['testing_feature'] = feature

    # save features
    save(savepath, feat) 

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass