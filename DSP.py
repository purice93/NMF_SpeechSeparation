# coding=utf-8
import pylab
import librosa
import librosa.output
import numpy as np
import wave as wav
from numpy.lib import stride_tricks
from sklearn.preprocessing import normalize


def merge(speaker, spec_dic):
    """针对speaker列表中的每一个人，合并得到谱图

    :param speaker: 说话人列表
    :param spec_dic: 幅度谱字典

    :return:若干个字典矩阵形成的数组
    """
    keys = spec_dic.keys()
    res = []
    for s in speaker:
        D = []
        for i in filter(lambda  x: s in x, keys):
            D.append(spec_dic[i])
        res.append(np.column_stack(D)) # 将array格式化
    return res

def getSpec(path):
    # file是时间-幅度序列（一维），75000；sample_rate是采样率，一秒钟的采样次数；音频为5秒
    file, sample_rate = librosa.load(path, sr=None) # 音频时间序列、采样率
    N = ( 32 *sample_rate ) //1000 # 双杠取整、单杠取余；N位帧长，为什么取这个值，不知道，求问？
    # 短时傅里叶变换：Short-time Fourier transform (STFT)
    # 又不懂了：为什么变换后的矩阵是长度是n_fft/2，而不是n_fft。n_fft不是帧长吗？
    return librosa.stft(file, n_fft=N, hop_length=N//2) # 帧长n_fft，帧移hop_length

# 重构语音
def reconstruct(f, mask,path):
    file, sample_rate = librosa.load(f, sr=None)
    N = (32 * sample_rate) // 1000
    mix = librosa.stft(file, n_fft=N, hop_length=N // 2)
    # 这里使用了维纳滤波的合成公式
    speaker = np.array(mix) * np.array(mask)
    sig = librosa.istft(speaker, hop_length=N//2)
    librosa.output.write_wav(path, sig, sample_rate)

# 重构语音
def reconstruct2(f, x1,path):
    file, sample_rate = librosa.load(f, sr=None)
    N = (32 * sample_rate) // 1000
    mix = librosa.stft(file, n_fft=N, hop_length=N // 2)
    # 这里使用了维纳滤波的合成公式
    print(np.shape(x1))
    print(N//2)
    sig = librosa.istft(x1, hop_length=N // 2)
    librosa.output.write_wav(path, sig, sample_rate)

# 重构语音,输出语音的频谱矩阵STFT
def getX1(f, mask):
    file, sample_rate = librosa.load(f, sr=None)
    N = (32 * sample_rate) // 1000
    mix = librosa.stft(file, n_fft=N, hop_length=N // 2)
    # 这里使用了维纳滤波的合成公式
    speaker = np.array(mix) * np.array(mask)
    return speaker