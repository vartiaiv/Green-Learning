import os
import pickle

def join_path_common(abs_path, rel_path):
    ret = None

    abs_path = os.path.normpath(abs_path)
    rel_path = os.path.normpath(rel_path)
    abs_s: list = abs_path.split(os.path.sep)
    rel_s: list = rel_path.split(os.path.sep)
    
    for i, a in enumerate(abs_s):
        if a == rel_s[0]:  # found common name
            abs_s = abs_s[0:i]  # keep until common name
            ret = os.path.normpath("/".join(abs_s + rel_s))
            break
    return ret


def mkdir_new(newpath):
    if os.path.exists(newpath):
        # print(f'Directory {newpath} exists already.')
        return
    os.makedirs(newpath)

def load(loadpath):
    # load data
    with open(loadpath, 'rb') as fr:
        data = pickle.load(fr, encoding='latin1')
    return data

def save(savepath, data):
    # save data
    dirpath, _ = os.path.split(savepath)
    mkdir_new(dirpath)  # make the save directory if needed
    with open(savepath, 'wb') as fw:
        pickle.dump(data, fw)

def load_params(modelpath, dataname):
    loadpath = os.path.join(modelpath, dataname)
    return load(loadpath)

def save_params(modelpath, dataname, params):
    savepath = os.path.join(modelpath, dataname)
    save(savepath, params)
