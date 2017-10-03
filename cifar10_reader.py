import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

def read_data():
    data_list = []
    labels_list = []
    for i in range(1,6):
        file = os.path.join('cifar10','data_batch_1')
        with open(file, 'rb') as fo:
            dicts = pickle.load(fo, encoding='bytes')
        data_list.append(dicts[b'data'])
        labels_list.append(dicts[b'labels'])
    X_train = np.concatenate(data_list)
    X_train = np.reshape(X_train, (len(X_train), 32 ,32 ,3)).astype(np.float32)
    y_train = np.concatenate(labels_list)
    y_train = np.eye(10)[y_train].astype(np.uint)
    
    
    file = os.path.join('cifar10','test_batch')
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    X_test = dicts[b'data'].astype(np.float32)
    X_test = np.reshape(X_test, (len(X_test), 32 ,32 ,3)).astype(np.float32)
    
    y_test = np.array(dicts[b'labels'])
    y_test = np.eye(10)[y_test].astype(np.uint)
    
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.
    
    return X_train, y_train, X_test, y_test