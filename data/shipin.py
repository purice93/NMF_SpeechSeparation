""" 
@author: zoutai
@file: shipin.py 
@time: 2017/12/28 
@description: 
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def transPngForAll(filepath):
    pathDir = os.listdir(filepath)
    for filename in pathDir:
        plotPng(filepath,filename)

def plotPng(filepath,filename):
    y, sr = librosa.load(filepath + filename, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))
    times = librosa.frames_to_time(beats, sr=sr)
    y_beat_times = librosa.clicks(times=times, sr=sr)

    # Or with a click frequency of 880Hz and a 500ms sample
    y_beat_times880 = librosa.clicks(times=times, sr=sr,
                                     click_freq=880, click_duration=0.5)

    # Display click waveform next to the spectrogram
    plt.figure()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    ax = plt.subplot(1, 1, 1)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             x_axis='time', y_axis='mel')
    # plt.subplot(2, 1, 1, sharex=ax)
    # librosa.display.waveplot(y_beat_times, sr=sr, label='Beat clicks')
    plt.legend()
    plt.xlim(0, 4)
    plt.tight_layout()

    portion = os.path.splitext(filename)
    # os.rename(filename,portion[0]+".png")

    # 切换文件路径,如无路径则要新建或者路径同上。否则会直接生成在当前文件夹下
    os.chdir("F:/研究生记录/研一/语音分离/一阶段/语音分离结果/样本4/")
    plt.savefig(portion[0]+".png")

path="F:/研究生记录/研一/语音分离/一阶段/语音分离结果/样本4/"
transPngForAll(path)