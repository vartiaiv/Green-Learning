import data
import saab

import pickle
import matplotlib.pyplot as plt

def getfeature():
    # load data
    with open('pca_params.pkl','rb') as fr:
        pca_params = pickle.load(fr, encoding='latin1')

    # read data
    train_images, _, test_images, _, _ = data.import_data("0-9")
    print('Training image size:', train_images.shape)
    print('Testing image size:', test_images.shape)

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

    with open('feat.pkl','wb') as fw:
        pickle.dump(feat, fw)    

if __name__ == "__main__":
    getfeature()