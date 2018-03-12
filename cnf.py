import tensorflow as tf

# 字典特征值Rank
#Rank = 70
#Rank = 170
Rank = 70

# 缓冲字典特征值设置为5
buffer = 5

# path = "./"
path = "./"

num_core = 2

def getConfig():
    """获取tensorFlow的配置
    
    :return: 配置
    """
    return tf.ConfigProto(inter_op_parallelism_threads=num_core,
                   intra_op_parallelism_threads=num_core)