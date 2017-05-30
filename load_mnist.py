import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "./dataset"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print("dataset must be 'testing' or 'training'")
        exit()

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    #get_img = lambda idx: (lbl[idx], img[idx])
    final_img = np.zeros((len(lbl),785))#784+1 bias
    for i in range(len(lbl)):
        final_img[i,:-1] = img[i].reshape((784))/255.0 #normalization
        final_img[i,-1] = 1
        if lbl[i]%2 == 0:
            lbl[i] = 1
        else:
            lbl[i] = -1

    final_lbl = np.expand_dims(lbl,axis=1)

    if dataset=="training":

        val_start = int(len(lbl)*4/5.0)
        return final_img[:val_start],final_lbl[:val_start],final_img[val_start:],final_lbl[val_start:]
    elif dataset=="testing":
        return final_img,final_lbl
    else:
        print("dataset must be 'testing' or 'training'")
        exit()



