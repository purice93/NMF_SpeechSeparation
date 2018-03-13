""" 
@author: zoutai
@file: testMore.py 
@time: 2018/03/13 
@description: 
"""
import tensorflow as tf

import cnf
from DSP import *
from fileManager import find_files
from tfnmf import TFNMF


import scipy.io as sio
from sklearn.decomposition import NMF

from DSP import *
from fileManager import find_files

Mix = getSpec("mix/tsp_speech_separation_mixture.wav")

# abs-计算整数、浮点数、复数的绝对值；这里是复数
Mix = np.abs(Mix)

data1 = sio.loadmat("h0.mat")
# w1 = data['w1'] # mat类型
H = data1['h0']  # <class 'numpy.ndarray'>

H1 = H[:70]
H2 = H[19:80]


data1 = sio.loadmat("r1.mat")
data2 = sio.loadmat("r2.mat")
data0 = sio.loadmat("r0.mat")
w1Set = data1['r1']  # <class 'numpy.ndarray'>
w2Set = data2['r2']  # <class 'numpy.ndarray'>
w0Set = data0['r0']  # <class 'numpy.ndarray'>

print(np.shape(w1Set))
# 为每个说话人拆分系数矩阵（两个人）
w1S1 = np.concatenate((np.mat(w1Set).T, (np.mat(w0Set).T))).T
w2S2 = np.concatenate((np.mat(w0Set).T, (np.mat(w2Set).T))).T
# w1S1 = w1Set.append(w0Set)
# w2S2 = w0Set.append(w2Set)

res1 = np.dot(w1S1, H1)  # X=WH
res2 = np.dot(w2S2, H2)  # X=WH
print("重构背景音和讲话人幅度谱")
#
# 更改：恢复相位
# 没法直接逆变换回去，所以还需要生成一个mask，这个操作就是假设相位相同
mask1 = res1 / Mix  # 一个浮点掩蔽
print(np.shape(mask1))

# # 先不重构，进行相位恢复
# x1 = getX1("mix/tsp_speech_separation_mixture.wav", mask1)
# print(x1)
# # 迭代求解，恢复相位
# for i in range(1000):
#     x1Mat = np.asmatrix(x1)
#     y1Mat = x1Mat * x1Mat.I
#     y1 = np.asarray(y1Mat)
#     x1 = np.dot(res1 / np.abs(y1), y1)
signal1 = reconstruct("mix/tsp_speech_separation_mixture.wav", mask1, "./doubleDic/test7_man.wav")

# 原始
mask2 = res2 / Mix  # 一个浮点掩蔽
signal2 = reconstruct("mix/tsp_speech_separation_mixture.wav", mask2, "./doubleDic/test7_woman.wav")
print("保存重构信号为音频文件")


# 保存日志
# 注意Windows环境下的的路径为了方便可以使用r''，或把\都替换成/或\\
# writer = tf.summary.FileWriter(r'F:\Users\log3', tf.get_default_graph())
# writer.close()