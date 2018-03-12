import wave
import numpy
import pylab as pl

def plotWav(f):
    # # 打开wav文件
    # # open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据
    # f = wave.open(r'mix.wav', 'rb')

    # 读取格式信息
    # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
    # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    # 读取波形数据
    # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_data = f.readframes(nframes)
    f.close()

    # 将波形数据转换成数组
    # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = numpy.fromstring(str_data, dtype=numpy.short)
    #wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = numpy.arange(0, nframes) * (1.0 / framerate)
    len_time = len(time)
    time = time[0:len_time]

    ##print "time length = ",len(time)
    ##print "wave_data[0] length = ",len(wave_data[0])

    # 绘制波形

    pl.subplot(111)
    pl.plot(time, wave_data)
    pl.xlabel("time")
    pl.show()