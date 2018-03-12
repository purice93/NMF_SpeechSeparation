#!/usr/bin/python
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF

def main():
    #user setting parameters
    V = np.random.rand(10000,10000)
    rank = 10
    num_core = 2

    tfnmf = TFNMF(V, rank)
    config = tf.ConfigProto(inter_op_parallelism_threads=num_core,
                                       intra_op_parallelism_threads=num_core)
    with tf.Session(config=config) as sess:
        start = time.time()
        W, H = tfnmf.run(sess)
        print("Computational Time for TFNMF: ", time.time() - start)

    W = np.mat(W)
    H = np.mat(H)
    error = np.power(V - W * H, 2).sum()
    print("Reconstruction Error for TFNMF: ", error)

if __name__ == '__main__':
    main()
