# from os import getcwd
import os
import pickle

def mkdir_new(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def load_pkl(dirpath, filename):
    # load data
    name, ext = filename.split('.')
    ext = "pkl" if ext == "pkl" else fr"{ext}.pkl"
    with open(fr"{dirpath}/{name}.{ext}",'rb') as fr:
        data = pickle.load(fr, encoding='latin1')
    return data

def write_pkl(dirpath, filename, data):
    name, ext = filename.split('.')
    mkdir_new(dirpath)
    ext = "pkl" if ext == "pkl" else fr"{ext}.pkl"
    with open(fr"{dirpath}/{name}.{ext}",'wb') as fw:
        pickle.dump(data, fw)