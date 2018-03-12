""" 
@author: zoutai
@file: shipin.py 
@time: 2017/12/28 
@description: 
"""
import numpy as np
import librosa
# 1. Get the file path to the included audio example
filename =filepath+'SA1.WAV'
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename,sr=None)
# load默认的采样率是22050，如果需要读取原始采样率,需要.load(filename,sr=None)而不是load(filename)
# 例如读取一段音频，判断节奏，并画出时频特性：

# Beat tracking example
#from __future__ import print_function
import librosa
import matplotlib.pyplot as plt
import librosa.display

# 1. Get the file path to the included audio example
# Sonify detected beat events
y, sr = librosa.load(librosa.util.example_audio_file())
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
y_beats = librosa.clicks(frames=beats, sr=sr)

# Or generate a signal of the same length as y
y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))

# Or use timing instead of frame indices
times = librosa.frames_to_time(beats, sr=sr)
y_beat_times = librosa.clicks(times=times, sr=sr)

# Or with a click frequency of 880Hz and a 500ms sample
y_beat_times880 = librosa.clicks(times=times, sr=sr,
                                 click_freq=880, click_duration=0.5)

# Display click waveform next to the spectrogram
plt.figure()
S = librosa.feature.melspectrogram(y=y, sr=sr)
ax = plt.subplot(2,1,2)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         x_axis='time', y_axis='mel')
plt.subplot(2,1,1, sharex=ax)
librosa.display.waveplot(y_beat_times, sr=sr, label='Beat clicks')
plt.legend()
plt.xlim(15, 30)
plt.tight_layout()