import matplotlib.pyplot as plt
import audioread
import wave
fig = plt.figure()

fig.add_subplot()

f = wave.open(r's1/tsp_speech_separation_groundtruth_male.wav','rb')
plt.title("Night.wav's Frames")
wavdata1 = f.readframes(4)
print(wavdata1)
plt.subplot(511)
plt.plot(14,wavdata1,color = 'green')

wavdata2,wavtime2 = wave.Wave_read('s2/mir1k_fdps_4_03_groundtruth_singing.wav')
plt.title("Night.wav's Frames")
plt.subplot(512)
plt.plot(wavtime2, wavdata2[0],color = 'green')

wavdata3,wavtime3 = wave.Wave_read('mix.wav')
plt.title("Night.wav's Frames")
plt.subplot(513)
plt.plot(wavtime3, wavdata3[0],color = 'green')

wavdata4,wavtime4 = wave.Wave_read('test.wav')
plt.title("Night.wav's Frames")
plt.subplot(514)
plt.plot(wavtime4, wavdata4[0],color = 'green')

plt.show()

