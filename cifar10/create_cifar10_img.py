#!/usr/bin/env python

import os
import numpy as np
import pickle
import cv2

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding = 'latin1') # different between notebook and .py
    return data_dict
def get_train_list(data_dir):
    train_list = []
    data_list = os.listdir(data_dir)
    for file_name in data_list:
        if 'data_batch' in file_name:
            train_list.append(file_name)
    return train_list
def get_test_list(data_dir):
    test_list = []
    data_list = os.listdir(data_dir)
    for file_name in data_list:
        if 'test_batch' in file_name:
            test_list.append(file_name)
    return test_list
def cifar10_to_png(data_dir, data_list, save_dir, set_type, label_name_list):
    save_root = os.path.join(save_dir, set_type) # train or test
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for label_name in label_name_list:
        class_root = os.path.join(save_root, label_name)
        if not os.path.exists(class_root):
            os.makedirs(class_root)
    for data_name in data_list:
        data_path = os.path.join(data_dir, data_name)
        xtr = unpickle(data_path)
        print(data_name + " is loading...")
        data = xtr['data']
        filenames = xtr['filenames']
        labels = xtr['labels']
        assert(data.shape[0] == len(filenames))
        assert(len(filenames) == len(labels))
        for idx, name in enumerate(filenames):
            img = np.reshape(data[idx], (3, 32, 32)).transpose(1, 2, 0)
            save_path = os.path.join(save_root, label_name_list[labels[idx]], name)
            cv2.imwrite(save_path, img)
if __name__ == "__main__":
    data_dir = 'cifar-10-batches-py'
    train_list = get_train_list(data_dir)
    test_list = get_test_list(data_dir)
    label_name_list = ['airplane',
                       'automobile',
                       'bird',
                       'cat',
                       'deer',
                       'dog',
                       'frog',
                       'horse',
                       'ship',
                       'truck']
    save_dir = 'cifar10'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # create train set
    cifar10_to_png(data_dir, train_list, save_dir, 'train', label_name_list)
    # create test set
    cifar10_to_png(data_dir, test_list, save_dir, 'test', label_name_list)
    