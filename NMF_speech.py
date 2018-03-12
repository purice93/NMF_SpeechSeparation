import tensorflow as tf

import cnf
from DSP import *
from fileManager import find_files
from tfnmf import TFNMF
import scipy.io as sio

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

    # 第二步
    # http://blog.csdn.net/JinbaoSite/article/details/73928729
    # 给定一个矩阵v，能够找到两个矩阵W、H满足，W*H=v
    # 其中W位语音特征，H为特征系数
    print("为各说话人训练字典")
    i=1
    for v in V:
        tf.reset_default_graph()
        tfnmf = TFNMF(v, cnf.Rank)  # 这里使用TensorFlow定义了一个乘法迭代的过程,cnf.Rank指特征个数
        with tf.Session(config=cnf.getConfig()) as sess:
            W, H = tfnmf.run(sess)
            # 保存中间字典值
            sio.savemat("w.mat",{"w"+str(i):np.mat(W)})
            i=i+1
        D.append(np.mat(W))
        print("完成一次矩阵分解,得到W形状：")
        print(W.shape)

    # 以下开始分离步骤
    print("读取混合音频,得到谱图形状:")
    Mix = getSpec("mix/tsp_speech_separation_mixture.wav")

    # abs-计算整数、浮点数、复数的绝对值；这里是复数
    Mix = np.abs(Mix)
    print(Mix.shape)



    # # 2018/01/20修改：加入公共字典sub-dic--T=B*W
    #
    # # --开始节点
    #
    # # 添加默认的缓冲子字典B
    # # scale uniform random with sqrt(V.mean() / rank)
    # nD = cnf.Rank * 2 + cnf.buffer
    # scale = 2 * np.sqrt(Mix.mean() / nD)
    # initializer = tf.random_uniform_initializer(maxval=scale)
    #
    # Mix = tf.constant(Mix, dtype=tf.float32)
    # shape = Mix.shape
    #
    # print("B is:",Mix.shape[0], cnf.buffer)
    # B = tf.get_variable("B", [Mix.shape[0], cnf.buffer],
    #                     initializer=initializer)
    #
    # print(B.shape)
    # print(type(B),type(D[0]))
    # # column_stack将拼接矩阵转化为二维矩阵
    # D.append(np.mat(B))
    #
    # # 先默认，循环10次（不知道是不是收敛的，但是太耗时，所以默认10次，估计也得几十个小时吧）
    # for k in range(10):
    #     W = np.column_stack(D)
    #     tf.reset_default_graph()
    #     print("开始矩阵分解")  # 分解为以W为特征的系数H
    #     tfnmf = TFNMF(Mix, cnf.Rank * 2 + cnf.buffer, algo="mud", D=W)  # 这里的调用不同，将之前获得的W传递进入算法
    #     with tf.Session(config=cnf.getConfig()) as sess:
    #         _, H = tfnmf.run(sess)
    #         print("得到激活矩阵H:")
    #         print(H.shape)
    #     print("开始重构信号:")
    #     m, n = H.shape
    #     # 为每个说话人拆分系数矩阵（两个人）
    #     # H1 = H[:m // 2]
    #     # H2 = H[m // 2:]
    #     H1 = H[:cnf.Rank]
    #     H2 = H[cnf.Rank:2 * cnf.Rank]
    #     H0 = H[2 * cnf.Rank:]
    #
    #     buffer = np.dot(D[2], H0)
    #     Mix += buffer
    #
    # res1 = np.dot(D[0], H1)  # X=WH
    # res2 = np.dot(D[1], H2)  # X=WH
    # print("重构背景音和讲话人幅度谱" + "---第" + k + "次")
    #
    # # --结束节点





    W = np.column_stack(D)
    tf.reset_default_graph()
    print("开始矩阵分解")  # 分解为以W为特征的系数H
    tfnmf = TFNMF(Mix, cnf.Rank * 2, algo="mud", D=W)  # 这里的调用不同，将之前获得的W传递进入算法
    with tf.Session(config=cnf.getConfig()) as sess:
        _, H = tfnmf.run(sess)
        print("得到激活矩阵H:")
        print(H.shape)
    print("开始重构信号:")
    m, n = H.shape
    # 为每个说话人拆分系数矩阵（两个人）
    H1 = H[:m // 2]
    H2 = H[m // 2:]
    res1 = np.dot(D[0], H1)  # X=WH
    res2 = np.dot(D[1], H2)  # X=WH
    print("重构背景音和讲话人幅度谱")
    #
    # 更改：恢复相位
    # 没法直接逆变换回去，所以还需要生成一个mask，这个操作就是假设相位相同
    mask1 = res1 / Mix # 一个浮点掩蔽
    # 先不重构，进行相位恢复
    x1 = getX1("mix/tsp_speech_separation_mixture.wav", mask1)
    print(x1)
    # 迭代求解，恢复相位
    for i in range(1000):
        x1Mat = np.asmatrix(x1)
        y1Mat = x1Mat * x1Mat.I
        y1 = np.asarray(y1Mat)
        x1 = np.dot(res1 / np.abs(y1),y1)
    signal1 = reconstruct2("mix/tsp_speech_separation_mixture.wav", x1,"./test6_man.wav")

    # 原始
    mask2 = res2 / Mix  # 一个浮点掩蔽
    signal2 = reconstruct("mix/tsp_speech_separation_mixture.wav", mask2, "./test6_woman.wav")
    print("保存重构信号为音频文件")


    # 保存日志
    # 注意Windows环境下的的路径为了方便可以使用r''，或把\都替换成/或\\
    # writer = tf.summary.FileWriter(r'F:\Users\log3', tf.get_default_graph())
    # writer.close()


if __name__ == '__main__':
    main()
