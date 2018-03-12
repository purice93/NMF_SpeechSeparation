""" 
@author: zoutai
@file: test171219.py 
@time: 2017/12/19 
@description: 测试混合信号是否为两个单独信号的叠加
"""

from DSP import getSpec
import numpy as np

'''
spec_dic = {}
f1 = "./s1/tsp_speech_separation_groundtruth_male.wav"
f2 = "./s2/tsp_speech_separation_groundtruth_female.wav"
spec1 = getSpec(f1) #获取谱的函数，实现见DSP文件（通过STFT）短时傅里叶变换
spec_dic[f1] = np.abs(spec1) # 返回帧t出处的频谱大小

spec2 = getSpec(f2)
spec_dic[f2] = np.abs(spec2)

# print("生成%s幅度谱"%(f1))
# print(spec_dic[f1])


mi = "./mix/tsp_speech_separation_mixture.wav"
specmi = getSpec(mi)
spec_dic[mi] = np.abs(specmi)
div = spec_dic[f1] +spec_dic[f2] - spec_dic[mi]
print(spec_dic[mi])
# print(div)

print(3.73478961+0.00569136-2.43984079)

'''

import pylab
import librosa
import librosa.output
import numpy as np
import wave as wav
from numpy.lib import stride_tricks
from sklearn.preprocessing import normalize
f1 = "./s1/tsp_speech_separation_groundtruth_male.wav"
f2 = "./s2/tsp_speech_separation_groundtruth_female.wav"
mi = "./mix/tsp_speech_separation_mixture.wav"
file1, sample_rate = librosa.load(f1, sr=None)  # 音频时间序列、采样率
file2, sample_rate = librosa.load(f2, sr=None)  # 音频时间序列、采样率
fileM, sample_rate = librosa.load(mi, sr=None)  # 音频时间序列、采样率
print(file1)
print(file2)
print(fileM)