""" 
@author: zoutai
@file: plot.py 
@time: 2017/11/09 
@description: 
"""
import plotme
import wave
f1 = wave.open(r's1/tsp_speech_separation_groundtruth_male.wav','rb')
plotme.plotWav(f1)
f2 = wave.open(r's2/tsp_speech_separation_groundtruth_female.wav','rb')
plotme.plotWav(f2)
# f3 = wave.open(r'test3.wav','rb')
# plotme.plotWav(f3)

