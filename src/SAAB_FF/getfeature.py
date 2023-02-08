import data
import saab

import pickle
import matplotlib.pyplot as plt

from utils.timer import timeit

@timeit
def main():
    # load data
    with open(r'./model/pca_params.pkl','rb') as fr:
        pca_params = pickle.load(fr, encoding='latin1')

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

    # save data

    with open(r'./model/feat.pkl','wb') as fw:
        pickle.dump(feat, fw)    

if __name__ == "__main__":
    main()