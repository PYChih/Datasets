#!/usr/bin/env python

#Source : https://github.com/teavanist/MNIST-JPG
import os
import struct
import sys
import numpy
from array import array
from os import path
from PIL import Image #imported from pillow
# funtion to read the MNIST dataset 
def read(dataset, path='mnist_download'):
    if dataset is "training":
        fname_img = "train-images.idx3-ubyte"
        fname_lbl = "train-labels.idx1-ubyte"
    elif dataset is "testing":
        fname_img = "t10k-images.idx3-ubyte"
        fname_lbl = "t10k-labels.idx1-ubyte"
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    fname_img = os.path.join(path, fname_img)
    fname_lbl = os.path.join(path, fname_lbl)

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols

# funtion to extract and  the MNIST dataset 
def write_dataset(labels, data, size, rows, cols, output_dir):
    output_dirs = [
        path.join(output_dir, str(i))
        for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".jpg")
        print("writing " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            data_array = numpy.asarray(data_i)


            im = Image.fromarray(data_array)
            im.save(output_filename)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        if not path.exists(output_path):
            os.makedirs(output_path)
    if len(sys.argv) > 2:
        input_path = sys.argv[2]
        if not path.exists(input_path):
            print('can not find input data...')
            sys.exit()
    else:
        if not path.exists('mnist_download'):
            print('setup your source data')
        else:
            input_path = 'mnist_download'

    for dataset in ["training", "testing"]:
        labels, data, size, rows, cols = read(dataset, input_path)
        write_dataset(labels, data, size, rows, cols,
                      path.join(output_path, dataset))