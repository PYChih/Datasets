#!/bin/bash
# Program:
#   convert mnist dataset from original file to jpg file
# prepare mnist original datafile include:
# t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte
# train-images.idx3-ubyte, train-labels.idx1-ubyte
# the folder structure is assumed to be:
# build_mnist.sh
# - <input_path>
#    - t10k-images.idx3-ubyte
#    - t10k-labels.idx1-ubyte
#    - train-images.idx3-ubyte
#    - train-labels.idx1-ubyte
# + <output_path>
#    + training
#       + 0
#           + 0.jpg...
#       + 1
#           + 1.jpg...
#       + 2
#        ...
#       + 9
#    + testing

echo "convert dataset to jpg..."
python3 create_mnist_jpg.py "$(pwd)"/mnist

echo "zip dataset for colab ..."
zip -r mnist.zip mnist/
