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
    print('--------Feature Extraction --------')
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")
    print("Model name:", os.path.basename(modelpath))
    
    use_num_features = FLAGS.use_num_features
    print("- USE_NUM_FEATURES:", use_num_features)

    # load train and test data
    train_images, train_labels, class_list = data_ffcnn.import_data()
    test_images, test_labels, _ = data_ffcnn.import_data(train=False)

    # select subset of features
    train_images, train_labels = \
        saab.select_balanced_subset(train_images, train_labels, use_num_features, class_list)

    print("Train images size:", train_images.shape)
    print("Test images size:", test_images.shape)

    # load trained PCA params for feature extraction
    pca_params = load_params(modelpath, "pca_params.pkl")

    # --------------------------------------------------------- training features
    print('--------Training features --------')
    train_feat = saab.initialize(train_images, pca_params)
    train_feat = train_feat.reshape(train_feat.shape[0], -1)
    # feature normalization
    std_var = (np.std(train_feat, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    train_feat = train_feat/std_var
    print("S4 training features shape:", train_feat.shape)
    # --------------------------------------------------------- testing features
    print('--------Testing features --------')
    test_feat = saab.initialize(test_images, pca_params)
    test_feat = test_feat.reshape(test_feat.shape[0], -1)
    # feature normalization
    std_var = (np.std(test_feat, axis=0)).reshape(1,-1)
    std_var[std_var == 0] = 1  # avoid divide by 0 error
    test_feat = test_feat/std_var
    print("S4 test features shape:", test_feat.shape)
    # ---------------------------------------------------------

    print('--------Finish Feature Extraction --------')

    # NOTE important! save these temporarily for getweight
    save_params(modelpath, "train_feat.pkl", train_feat)
    save_params(modelpath, "train_labels.pkl", train_labels)
    save_params(modelpath, "test_feat.pkl", test_feat)
    save_params(modelpath, "test_labels.pkl", test_labels)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass