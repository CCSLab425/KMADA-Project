# -*- coding: utf-8 -*-


import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def stft(x, **params):
    """
    :param x: 输入信号
    :param params: {fs:采样频率；
                    window:窗。默认为汉明窗；
                    nperseg： 每个段的长度，默认为256，
                    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                    nfft：fft长度，
                    detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                    return_onesided：默认为True，返回单边谱。
                    boundary：默认在时间序列两端添加0
                    padded：是否对时间序列进行填充0（当长度不够的时候），
                    axis：可以不必关心这个参数}
    :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
    """
    f, t, zxx = signal.stft(x, **params) 
    return f, t, zxx 


def stft_specgram(x, picname=None, **params):  # picname是给图像的名字，为了保存图像
    f, t, zxx = stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()
    if picname is not None:
        plt.savefig('..\\picture\\' + str(picname) + '.jpg')   # 保存图像
    plt.clf()   # 清除画布
    return t, f, zxx


t = np.arange(0,10.24, 0.01)
S = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)
t, f, zxx = stft_specgram(S)