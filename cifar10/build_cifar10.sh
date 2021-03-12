#!/bin/bash
# Program:
#   convert cifar10 dataset from original file to jpg file
# 2021/03/12 can download from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz


# wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar zxvf cifar-10-python.tar.gz

python3 create_cifar10_img.py

echo "zip dataset for colab ..."
zip -r cifar10.zip cifar10/