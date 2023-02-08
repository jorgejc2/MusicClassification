import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy
from IPython.display import Audio

import build.pybind_modules.dsp_module as cu
from time import time

def get_signal_Hz(Hz,sample_rate,length_ts_sec):
    ## 1 sec length time series with sampling rate 
    ts1sec = list(np.linspace(0,np.pi*2*Hz,sample_rate))
    ## 1 sec length time series with sampling rate 
    # multiplying a list by a constant appends constant copies to the list itself
    ts = ts1sec*length_ts_sec
    return(list(np.sin(ts)))

if __name__ == "__main__":

    sample_rate   = 4000
    length_ts_sec = 4
    ## --------------------------------- ##
    ## 3 seconds of "digit 1" sound
    ## Pressing digit 2 buttom generates 
    ## the sine waves at frequency 
    ## 697Hz and 1209Hz.
    ## --------------------------------- ##
    ts1  = np.array(get_signal_Hz(697, sample_rate,length_ts_sec)) 
    ts1 += np.array(get_signal_Hz(1209,sample_rate,length_ts_sec))
    ts1  = list(ts1)

    ## -------------------- ##
    ## 2 seconds of silence
    ## -------------------- ##
    ts_silence = [0]*sample_rate*1

    ## --------------------------------- ##
    ## 3 seconds of "digit 2" sounds 
    ## Pressing digit 2 buttom generates 
    ## the sine waves at frequency 
    ## 697Hz and 1336Hz.
    ## --------------------------------- ##
    ts2  = np.array(get_signal_Hz(697, sample_rate,length_ts_sec)) 
    ts2 += np.array(get_signal_Hz(1336,sample_rate,length_ts_sec))
    ts2  = list(ts2)

    ## -------------------- ##
    ## Add up to 7 seconds
    ## ------------------- ##
    # concatenates the lists, doesn't sum their values like a numpy array
    ts = ts1 + ts_silence  + ts2

    cu_start = time()
    result = cu.cuFFT(list(ts[:1024]))
    cu_end = time()
    print("cuFFT took {} ms to finish".format((cu_end - cu_start)*1000))

    np_start = time()
    np_result = np.fft.fft(ts[:1024])
    np_end = time()
    print("numpy's fft took {} ms to finish".format((np_end - np_start)*1000))

    results_close = np.allclose(result, np_result)
    print("Results match: {}\n".format(results_close))

    nfft = 256 
    noverlap = -1
    cu_start = time()
    result = cu.cuSTFT(list(ts), sample_rate, nfft, noverlap, True, 0)
    cu_end = time()
    print("cuSTFT took {} ms to finish".format((cu_end - cu_start)*1000))
    scipy_start = time()
    f, t, scipy_result = scipy.signal.stft(ts, fs=sample_rate, window="boxcar", nperseg=nfft, noverlap=None if noverlap == -1 else noverlap, nfft=nfft, detrend=False, return_onesided=True, boundary=None, padded=False, axis=-1, scaling='psd')
    # convert to power spectral density
    # since scipy_result is a complex spectrum, the equation to get PSD is 20*log10(abs(scipy_result))
    # https://dsp.stackexchange.com/questions/54811/how-to-calculate-the-psd-from-the-complex-calculated-stft
    scipy_result = np.abs(scipy_result)
    scipy_result = 20*np.log10(scipy_result)
    scipy_end = time()
    print("scipy's stft took {} ms to finish".format((scipy_end - scipy_start)*1000))
    result = result[:,:-1] # my stft always takes one more time slice than scipy stft does

    result_1D = result.ravel()
    scipy_result_1D = scipy_result.ravel()

    total = len(scipy_result_1D)
    mismatches = 0
    for i in range(len(scipy_result_1D)):
        if not np.isclose(scipy_result_1D[i], result_1D[i]):
            mismatches += 1

    percent_error = (mismatches / total) * 100
    print("cuSTFT error: {}%".format(percent_error))
