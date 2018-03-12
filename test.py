""" 
@author: zoutai
@file: test.py 
@time: 2017/12/14 
@description:
"""
import numpy as np
from numpy import *
# nu = ones((5,4))
# ma = np.asmatrix(nu)
# outp = ma * ma.I
# nup = np.asarray(outp)
# print(nup)
# print(outp)
from NMF_speech import *

# 测试1
# y = [[  1.73653495e+00, +0.00000000e+00j,  -8.82204622e-03, -0.00000000e+00j,
#    -1.56715028e-02, -0.00000000e+00j,    -2.25590752e-03, -0.00000000e+00j,
#    -2.14958540e-03, -0.00000000e+00j,  -9.40178754e-04, -0.00000000e+00j],
#  [ -1.99658465e+00, +1.38069163e-16j,  -1.00812986e-02, -5.51020121e-03j,
#     9.80248768e-03, -1.34509951e-02j,     8.23584385e-03, -6.70998823e-03j,
#     1.39904367e-02, -7.63005763e-03j,  -4.27376293e-03, -3.15120327e-03j],
#  [  2.79350996e+00, -1.97209913e-19j,   3.86687764e-03, +1.57662649e-02j,
#    -1.29563399e-02, +3.18123889e-03j,    -2.11877301e-02, -1.82316210e-02j,
#    -2.08487120e-02 ,+2.79168580e-02j,  -4.56183916e-03, +2.48002750e-03j],
#  [  1.35144182e-02, -8.12886085e-16j,  -7.13774860e-02, -9.03674960e-02j,
#     1.62311301e-01 ,-1.26290336e-01j,    -9.14356613e-04, -5.80456341e-04j,
#    -1.23975810e-03, +7.54498818e-04j,   3.96551986e-05, -4.36784030e-05j],
#  [  1.18997302e-02, +2.20007560e-17j,   1.17468433e-02, +1.03127681e-01j,
#    -1.84425563e-01, +2.16428097e-02j,     2.92067707e-04, +3.23708809e-04j,
#     4.22241836e-04, -7.23792473e-04j,   5.87326167e-06, +6.65949774e-05j],
#  [  8.45793914e-03, +0.00000000e+00j,   7.31404349e-02, +0.00000000e+00j,
#     1.35931984e-01, +0.00000000e+00j,    -4.25534789e-04, -0.00000000e+00j,
#    -7.36354035e-04, -0.00000000e+00j,  -3.84656705e-05, -0.00000000e+00j]]
# Mix = getSpec("mix/tsp_speech_separation_mixture.wav")
# Mix = np.abs(Mix)
# y1 = ones((257,257))
# y1[:][:] = 9.80248768e-03
# res1 = ones((257,814))
# res1[:][:] = 1
# print(np.dot(y1,res1))
# # print(np.abs(y1))
# x1 = res1 / linalg.det(y1)
# x1 = np.dot(y1,x1)
# rs = x1 / Mix
# # signal2 = reconstruct("mix/tsp_speech_separation_mixture.wav", rs,"./test7_woman.wav")
# # #signal1 = reconstruct2("mix/tsp_speech_separation_mixture.wav", x1,"./test6_man.wav")
# # print("保存重构信号为音频文件")



# 测试2
# # 2018/03/11测试np.mat存储为文件
# import scipy.io as sio
#
#
# m1 = np.ones((2,2))
# print(type(m1))
# m2 = np.mat(m1)
# print(m2)
# sio.savemat("12.mat", {'w1':m2})
#
# mat1 = sio.loadmat("12.mat")
# data1 = mat1['w1']
# print("122",data1)
#
# # 完成测试



# # 测试3sklearn的NMF
# import numpy as np
# X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
# from sklearn.decomposition import NMF
# model = NMF(n_components=4, init='random', random_state=0)
# W = model.fit_transform(X)
# H = model.components_
# print(W)
# print(H)


# 测试4-测试矩阵分解是否有无穷个解


# import numpy as np
# X = np.mat([[11], [21], [33], [45], [534], [62]])
# w = np.mat([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
# h = w.I*X
# print(h)
# print(np.rank(h))



w1Set = [[1,21,34],[32,14,5]]
print(np.shape(w1Set)[0])
