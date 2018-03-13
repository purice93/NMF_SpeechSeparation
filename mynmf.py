""" 
@author: zoutai
@file: mynmf.py 
@time: 2018/03/11 
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


def main():
    '''
    主函数，包含分离的主要过程
    :return:
    '''
    speaker = ["s1", "s2", ]  # 这里是说话人的数据文件夹名称
    spec_dic = {}
    for s in speaker:
        files = find_files(s)  # 自己写的一个便利文件的工具
        print("完成%s相关文件读取" % (s))
        for f in files:
            print(f)
            spec = getSpec(f)  # 获取谱的函数，实现见DSP文件（通过STFT）短时傅里叶变换
            print("生成%s幅度谱" % (f))
            spec_dic[f] = np.abs(spec)  # 返回帧t出处的频谱大小

    V = merge(speaker, spec_dic)  # 把这些谱拼成一个大的准备分解（是否拼接音频文件再转换成谱更好）
    del (spec_dic)  # 这里是去除掉之前一步的中间变量，如果数据量大，整个过程很费内存
    print("清理内存")
    D = []  # 保存语音特征矩阵
    i = 1
    for v in V:
        model = NMF(n_components=70, init='random', random_state=0)
        W = model.fit_transform(v)
        H = model.components_
        # 保存中间字典值
        sio.savemat("myw" + str(i) + ".mat", {"w" + str(i): np.mat(W)})
        i = i + 1

    data1 = sio.loadmat("myw1.mat")
    # w1 = data['w1'] # mat类型
    w1 = data1['w1'] # <class 'numpy.ndarray'>
    m, n = w1.shape  # 257 70 # 没有括号
    print("hello", w1[:, 1])

    data1 = sio.loadmat("myw2.mat")
    # w1 = data['w1'] # mat类型
    w2 = data1['w2']# <class 'numpy.ndarray'>
    m, n = w2.shape  # 257 70 # 没有括号
    print("hello", w2[:, 1])

    w1Set = []
    w2Set = []
    w0Set = []
    wSetAll = []

    # 分别分解w1,w2
    myarr = np.arange(0, n, 1).tolist()
    for i in range(n):
        # model1 = NMF(n_components=70, init='random', random_state=0)
        # W = model.fit_transform(w1[:,i])
        # H = model.components_
        # 直接采用矩阵分解
        # print(w1.shape)
        # print(np.mat((w1[:, i])).T.shape)
        H = np.mat(w2).I * np.mat((w1[:, i])).T
        # 判断H中是否存在负值，如果存在，则不属于内部字典，即属于另外一个字典的特有字典
        print("std",np.std(H)) # 求标准差
        if np.std(H) > 1:
            w1Set.append(np.array(w1[:, i]))
            # np.row_stack((w2Set, np.array(w1[:, i])))  # 添加一行数据
            myarr.remove(i)
        # for j in range(n):
        #     if H[j][0] < 0:
        #         w1Set.append(np.array(w1[:, i]))
        #         # np.row_stack((w2Set, np.array(w1[:, i])))  # 添加一行数据
        #         myarr.remove(i)
        #         break

    for i in range(n):
        # model1 = NMF(n_components=70, init='random', random_state=0)
        # W = model.fit_transform(w2[:,i])
        # H = model.components_
        H = np.mat(w1).I * np.mat((w2[:, i])).T
        # 判断H中是否存在负值，如果存在，则不属于内部字典，即属于另外一个字典的特有字典

        if np.std(H) > 1:
            w2Set.append(np.array(w2[:, i]))
            # np.row_stack((w2Set, np.array(w1[:, i])))  # 添加一行数据
        # for j in range(n):
        #     if H[j][0] < 0:
        #         w2Set.append(np.array(w2[:, i]))
        #         break

    for k in range(len(myarr)):
        w0Set.append(np.array(w1[:, myarr[k]]))

    sio.savemat("r1.mat", {"r1": np.mat(w1Set).T})
    sio.savemat("r2.mat", {"r2": np.mat(w2Set).T})
    sio.savemat("r0.mat", {"r0": np.mat(w0Set).T})

    wSetAll.extend(w1Set)
    wSetAll.extend(w0Set)
    wSetAll.extend(w2Set)
    print(np.shape(w1Set)[1])
    # modelAll = NMF(n_components=(np.shape(w1Set)[1]+np.shape(w0Set)[1]+np.shape(w2Set)[1]), init='random', random_state=0)
    # modelAll = NMF(n_components=(np.shape(w1Set)[1]+np.shape(w2Set)[1]), init='random', random_state=0)



    # 以下开始分离步骤
    print("读取混合音频,得到谱图形状:")
    Mix = getSpec("mix/tsp_speech_separation_mixture.wav")

    # abs-计算整数、浮点数、复数的绝对值；这里是复数
    Mix = np.abs(Mix)
    print(Mix.shape)

    # W = np.column_stack(D)
    tf.reset_default_graph()
    print("开始矩阵分解")  # 分解为以W为特征的系数H
    print(np.shape(wSetAll))
    print(np.shape(w1Set))
    print(np.shape(w2Set))
    print(np.shape(w0Set))
    W = np.column_stack(wSetAll)
    W = tf.cast(W,tf.float32)  # Input 'b' of 'MatMul' Op has type float32 that does not match type float64 of argument 'a'.
    print(type(Mix[0][0]),type(W[0][0]))
    colLen = (np.shape(w1Set)[0]+np.shape(w0Set)[0]+np.shape(w2Set)[0])
    tfnmf = TFNMF(Mix, colLen, algo="mud", D=W)  # 这里的调用不同，将之前获得的W传递进入算法
    with tf.Session(config=cnf.getConfig()) as sess:
        _, H = tfnmf.run(sess)
        print("得到激活矩阵H:")
        print(H.shape)
    print("开始重构信号:")
    m, n = H.shape
    # 为每个说话人拆分系数矩阵（两个人）
    w1S1 = w1Set.append(w0Set)
    w2S2 = w0Set.append(w2Set)
    try:
        sio.savemat("h0.mat", {"h0": np.mat(H)})
    except Exception:
        print("save error")

    H1 = H[:np.shape(w1Set)[0]+np.shape(w0Set)[0]]
    H2 = H[np.shape(w1Set)[0]+np.shape(w0Set)[0]:np.shape(w1Set)[0]+np.shape(w0Set)[0]+np.shape(w2Set)[0]]
    res1 = np.dot(w1S1, H1)  # X=WH
    res2 = np.dot(w2S2, H2)  # X=WH
    print("重构背景音和讲话人幅度谱")
    #
    # 更改：恢复相位
    # 没法直接逆变换回去，所以还需要生成一个mask，这个操作就是假设相位相同
    mask1 = res1 / Mix  # 一个浮点掩蔽

    # # 先不重构，进行相位恢复
    # x1 = getX1("mix/tsp_speech_separation_mixture.wav", mask1)
    # print(x1)
    # # 迭代求解，恢复相位
    # for i in range(1000):
    #     x1Mat = np.asmatrix(x1)
    #     y1Mat = x1Mat * x1Mat.I
    #     y1 = np.asarray(y1Mat)
    #     x1 = np.dot(res1 / np.abs(y1), y1)
    signal1 = reconstruct2("mix/tsp_speech_separation_mixture.wav", mask1, "./test7_man.wav")

    # 原始
    mask2 = res2 / Mix  # 一个浮点掩蔽
    signal2 = reconstruct("mix/tsp_speech_separation_mixture.wav", mask2, "./test7_woman.wav")
    print("保存重构信号为音频文件")


    # 保存日志
    # 注意Windows环境下的的路径为了方便可以使用r''，或把\都替换成/或\\
    # writer = tf.summary.FileWriter(r'F:\Users\log3', tf.get_default_graph())
    # writer.close()

if __name__ == '__main__':
    main()
