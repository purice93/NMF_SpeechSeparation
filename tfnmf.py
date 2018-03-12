
from __future__ import division
import numpy as np
import tensorflow as tf

INFINITY = 10e+12

class TFNMF(object):
    def __init__(self, V, rank, algo="mu", learning_rate=0.01, D=None):

        #convert numpy matrix(2D-array) into TF Tensor
        self.V = tf.constant(V, dtype=tf.float32)
        shape = V.shape

        self.rank = rank
        self.algo = algo
        self.lr = learning_rate

        #scale uniform random with sqrt(V.mean() / rank)
        scale = 2 * np.sqrt(V.mean() / rank)
        initializer = tf.random_uniform_initializer(maxval=scale)

        # 1.初始化H = （70 * n）,每个值为V的均值/70..;
        self.H =  tf.get_variable("H", [rank, shape[1]],
                                     initializer=initializer)
        # 2.初始化W = (m * 70),值同上
        if algo == "mud":
            self.W = tf.Variable(D,name="W")
        else:
            self.W =  tf.get_variable(name="W", shape=[shape[0], rank],
                                     initializer=initializer)

        if algo == "mu":
            self._build_mu_algorithm()
        elif algo == "mud":
            self._build_mud_algorithm()
        else:
            raise ValueError("The attribute algo must be in {'mu', 'mud'}")

    def _build_mu_algorithm(self):
        """build dataflow graph for Multiplicative algorithm"""

        V, H, W = self.V, self.H, self.W
        rank = self.rank
        shape = V.get_shape()

        graph = tf.get_default_graph()

        #save W for calculating delta with the updated W
        W_old = tf.get_variable(name="W_old", shape=[shape[0], rank])
        save_W = W_old.assign(W)

        #Multiplicative updates
        with graph.control_dependencies([save_W]):
            #update operation for H
            Wt = tf.transpose(W)
            WV = tf.matmul(Wt, V)
            WWH = tf.matmul(tf.matmul(Wt, W), H)
            WV_WWH = WV / WWH
            #select op should be executed in CPU not in GPU
            with tf.device('/cpu:0'):
                #convert nan to zero
                WV_WWH = tf.where(tf.is_nan(WV_WWH),
                                    tf.zeros_like(WV_WWH),
                                    WV_WWH)
            H_new = H * WV_WWH
            update_H = H.assign(H_new)

        with graph.control_dependencies([save_W, update_H]):
            #update operation for W (after updating H)
            Ht = tf.transpose(H)
            VH = tf.matmul(V, Ht)
            WHH = tf.matmul(W, tf.matmul(H, Ht))
            VH_WHH = VH / WHH
            with tf.device('/cpu:0'):
                VH_WHH = tf.where(tf.is_nan(VH_WHH),
                                        tf.zeros_like(VH_WHH),
                                        VH_WHH)
            W_new = W * VH_WHH
            update_W = W.assign(W_new)

        self.delta = tf.reduce_sum(tf.abs(W_old - W))

        self.step = tf.group(save_W, update_H, update_W)

    def _build_mud_algorithm(self):
        """build dataflow graph for Multiplicative algorithm"""

        V, H, W = self.V, self.H, self.W
        rank = self.rank
        shape = V.get_shape()

        graph = tf.get_default_graph()

        # ？
        # save W for calculating delta with the updated W
        H_old = tf.get_variable(name="H_old", shape=[rank, shape[1]])
        save_H = H_old.assign(H)

        # 核心：WH的迭代过程计算
        # Multiplicative updates
        with graph.control_dependencies([save_H]):
            # update operation for H
            Wt = tf.transpose(W)
            WV = tf.matmul(Wt, V)
            WWH = tf.matmul(tf.matmul(Wt, W), H)
            WV_WWH = WV / WWH
            # select op should be executed in CPU not in GPU
            with tf.device('/cpu:0'):
                # convert nan to zero
                WV_WWH = tf.where(tf.is_nan(WV_WWH),
                                   tf.zeros_like(WV_WWH),
                                   WV_WWH)
            H_new = H * WV_WWH
            update_H = H.assign(H_new)

        self.delta = tf.reduce_sum(tf.abs(V-tf.matmul(W,H_new)))

        self.step = tf.group(save_H, update_H)

    def run(self, sess, max_iter=999999, min_delta=0.001):
        algo = self.algo

        tf.global_variables_initializer().run()

        if algo == "mu":
            return self._run_mu(sess, max_iter, min_delta)
        elif algo == "mud":
            return self._run_mud(sess, max_iter, min_delta)
        else:
            raise ValueError

    def _run_mud(self, sess, max_iter, min_delta):
        for i in range(max_iter):
            self.step.run()
            delta = self.delta.eval()
            if delta < min_delta:
                break
        W = self.W.eval()
        H = self.H.eval()
        return W, H

    def _run_mu(self, sess, max_iter, min_delta):
        for i in range(max_iter):
            self.step.run()
            delta = self.delta.eval()
            if delta < min_delta:
                break
        W = self.W.eval()
        H = self.H.eval()
        return W, H
