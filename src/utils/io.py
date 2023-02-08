import os
import pickle

def combine_with_duplicate(root, rel_path):
    """ Combine paths from common directory. Otherwise join normally.
    Examples:
    'the/path/to/project/src/file', 'project/data' combine into
    => 'the/path/to/project/data'
    'the/path/to/project/src/package', 'pkg1' combine into
    => 'the/path/to/project/data'
    """
    root = os.path.normpath(root)  # to system path separator, an escaped backslash '\\'
    rel_path = os.path.normpath(rel_path)

    rs = root.split(os.sep)
    rps = rel_path.split(os.sep)
    popped = False
    for v in rs:
        if v == rps[0]:
            rps.pop(0)
            popped = True
        elif popped:
            break
    return "/".join(rs+rps)


def mkdir_new(newpath):
    if os.path.exists(newpath):
        print(f'Directory {newpath} exists already.')
        return
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
