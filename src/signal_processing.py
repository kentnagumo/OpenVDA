import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import numpy as np
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)

TMP_DIR = 'result_tmp/'


def read_csv(file, header=0):
    logger.debug('enter')
    df = pd.read_csv(file, header=header, engine='python')
    logger.debug('exit')
    return df


def read_table(file, header=0):
    logger.debug('enter')
    df = pd.read_table(file, header=header, sep=',')
    logger.debug('exit')
    return df


def fir_filter(data, N, fs, cutoff, fi_type, plot=False):
    '''FIRフィルター
    
    Parameters
    ----------
    data: Numpy
        フィルターに通すデータ
    N: int
        次数
    fs: int
        サンプリング周波数
    cutoff: double or list
        帯域通過周波数, バンドパスフィルターフィルター時は、listで指定
    fi_type: str
        フィルターの種類: low, high, pass, stop
    plot: bool
        フィルター結果の図示の設定    
    
    Returns
    -------
    result: Numpy
        フィルター後のデータ
        
    '''
    fn = fs/2
    
    # pass_zeroの設定
    if fi_type == 'low' or fi_type == 'stop':
        pass_zero = True    
    elif fi_type == 'high' or fi_type == 'pass':
        pass_zero = False
        
    #FIRフィルターの関数の周波数値の設定
    if fi_type == 'low':
        fil_fs = cutoff/fn
    elif fi_type == 'high':
        fil_fs = cutoff/fn
    elif fi_type == 'stop' or fi_type == 'pass':
        fil_fs = [cutoff[0]/fn, cutoff[1]/fn]
    
    fir_middle = signal.firwin(N, fil_fs, window='hanning', pass_zero=pass_zero)
    result = signal.filtfilt(fir_middle, 1, data)
    
    if plot == True:
        # 振幅と位相の算出
        w, h = signal.freqz(fir_middle)
        frq = (w/np.pi)* fn
        
        # 利得ー周波数応答の図示
        plt.plot(frq  , 20 * np.log10(abs(h)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.title('Filter frequency response')
        plt.grid(which='both', axis='both')
        plt.show()

        # 位相ー周波数応答の図示
        plt.plot(frq  , np.angle(h)* 180 / np.pi)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [deg]')
        plt.title('Filter frequency response')
        plt.grid(which='both', axis='both')
        plt.show()
        
        # フィルターの結果の図示
        plt.plot(data, alpha=0.8)
        plt.plot(result, alpha=0.8)
        plt.show()
    
    return result
    
    
def iir_filter(data, order, fs, cutoff, fi_type, plot=False):
    '''IIRフィルター
    
    Parameters
    ----------
    data: Numpy
        フィルターに通すデータ
    order: int
        次数
    fs: int
        サンプリング周波数
    cutoff: double or list
        帯域通過周波数, バンドパスフィルターフィルター時は、listで指定
    fi_type: str
        フィルターの種類: low, high, pass, stop
    plot: bool
        フィルター結果の図示の設定    
    
    Returns
    -------
    result: Numpy
        フィルター後のデータ
        
    '''
    fn = fs/2
    
    if fi_type == 'low':
        fil_fs = cutoff/fn
    elif fi_type == 'high':
        fil_fs = cutoff/fn
    elif fi_type == 'stop' or fi_type == 'pass':
        fil_fs = [cutoff[0]/fn, cutoff[1]/fn]    
    
    b, a = signal.butter(order, fil_fs, fi_type)
    result = signal.filtfilt(b, a, data)
    
    if plot==True:
        # 周波数と利得の計算
        w, h = signal.freqz(b, a)
        frq = (w/np.pi)* fn

        # 利得ー周波数応答の図示
        plt.plot(frq  , 20 * np.log10(abs(h)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.title('Butterworth filter frequency response')
        plt.grid(which='both', axis='both')
        plt.show()

        # 位相ー周波数応答の図示
        plt.plot(frq  , np.angle(h)* 180 / np.pi)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [deg]')
        plt.title('Butterworth filter frequency response')
        plt.grid(which='both', axis='both')
        plt.show()
    
    return result
    
    
if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    # ログレベルのセット
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)
    
    handler = FileHandler(TMP_DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler) 
    
    logger.info('Start')
    # データの作成
    t = np.linspace(0, 10 ,200)
    sin_5hz = np.sin(2 * np.pi * 5 * t)
    sin_50hz = np.sin(2 * np.pi * 50 * t)
    sample_data = sin_5hz + sin_50hz
    
    logger.info('fir_filter start')
    iirdata = iir_filter(sample_data, 5, 20, 2, 'high', plot=False)
    logger.info('\n{}'.format(iirdata))
    
    logger.info('End')