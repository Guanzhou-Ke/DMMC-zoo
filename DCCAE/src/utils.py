import gzip
import os

import torch
from tqdm import tqdm
import requests
from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

from data import NoisyMnistDataset


def download_file(path, url):
    print(f"Downloading {url} to {path}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        chunk_size = 8192
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    print(f"OK! Saved at {path}.")
    return None

def make_tensor(data_xy):
    """converts the input to tensor"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x).float()
    data_y = torch.tensor(data_y)
    return data_x, data_y

def load_noisy_mnist(root='./data'):
    # Download dataset if not existence.
    file1, file2 = os.path.join(root, 'noisymnist_view1.gz'), os.path.join(root, 'noisymnist_view2.gz')
    if not os.path.exists(root):
        os.makedirs(root)
        
    if not os.path.exists(file1):
        download_file(file1, 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
    else:
        print("Noisy mnist view1 has existen!")
        
    if not os.path.exists(file2):
        download_file(file2, 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')
    else:
        print("Noisy mnist view2 has existen!")
    
    # Unzip and prepare dataset
    f = gzip.open(file1, 'rb')
    view1_train_set, view1_valid_set, view1_test_set = load_pickle(f)
    f.close()

    view1_train_set_x, view1_train_set_y = make_tensor(view1_train_set)
    view1_valid_set_x, view1_valid_set_y = make_tensor(view1_valid_set)
    view1_test_set_x, view1_test_set_y = make_tensor(view1_test_set)

    
    f = gzip.open(file2, 'rb')
    view2_train_set, view2_valid_set, view2_test_set = load_pickle(f)
    f.close()

    view2_train_set_x, view2_train_set_y = make_tensor(view2_train_set)
    view2_valid_set_x, view2_valid_set_y = make_tensor(view2_valid_set)
    view2_test_set_x, view2_test_set_y = make_tensor(view2_test_set)
    
    return (NoisyMnistDataset([view1_train_set_x, view2_train_set_x, view1_train_set_y]),
            NoisyMnistDataset([view1_valid_set_x, view2_valid_set_x, view1_valid_set_y]),
            NoisyMnistDataset([view1_test_set_x, view2_test_set_x, view1_test_set_y]))
    

def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def svm_classify(data, C=1.0):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    valid_data, valid_label = data[1]
    test_data, test_label = data[2]

    print('Training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())
    print('Done!')

    
    pred = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, pred)
    valid_nmi = normalized_mutual_info_score(valid_label, pred)
    
    pred = clf.predict(test_data)
    test_acc = accuracy_score(test_label, pred)
    test_nmi = normalized_mutual_info_score(test_label, pred)
    
    print(f"Validation accuracy: {valid_acc*100:.2f}%, test accuracy: {test_acc*100:.2f}%")
    print(f"Validation NMI: {valid_nmi*100:.2f}%, test NMI: {test_nmi*100:.2f}%")
    return ((valid_acc, test_acc), (valid_nmi, test_nmi))