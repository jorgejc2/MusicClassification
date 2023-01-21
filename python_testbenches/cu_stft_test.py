"""
Jorge Chavez

Testbench used to test CUDA code for cuFFT
"""

# set python file's path directory of modules 
import sys
sys.path.append('../build/pybind_modules')
# rest of imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os 
import dsp_module as cu

dir_path = os.path.dirname(os.path.realpath(__file__))

class testbench():
    
    # testbench results

    tb_results = {
        "ringtone_test" : False,
        "simple_one": False,
        "simple_two": False,
        "simple_three": False,
    }
    
    # helper functions

    def get_signal_Hz(self,Hz,sample_rate,length_ts_sec):
        """
        Description: Returns signal in Hz
        Inputs: 
            Hz -- frequency to replicate
            sample_rate -- signal rate of signal
            length_ts_sec -- length of the signal
        Ouputs: None
        Returns: list of values replicating the desired signal
        Effects: None
        """
        ## 1 sec length time series with sampling rate 
        ts1sec = list(np.linspace(0,np.pi*2*Hz,sample_rate))
        ## 1 sec length time series with sampling rate 
        # multiplying a list by a constant appends constant copies to the list itself
        ts = ts1sec*length_ts_sec
        return(list(np.sin(ts)))

    def get_psd(self, signal, fft_size, sample_rate):
        return 10.0 * np.log10( np.abs(signal)**2 / (fft_size*sample_rate) )

    # tests

    def simple_one(self, visualize_output: bool = False):
        """
        Description: Basic FFT with 32 numbers
        Inputs: 
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        
        # checking math on exponentials 
        FFT_size = 32

        test = np.array([i for i in range(FFT_size)])

        result = cu.cuFFT(list(test))
        correct_result = np.fft.fft(test)

        tb_result = np.allclose(result, correct_result)

        if visualize_output:
            print("Close: {}".format(tb_result))
            for i in range(len(correct_result)):
                print("correct: {}, your output: {}".format(correct_result[i], result[i]))

        self.tb_results["simple_one"] = tb_result

    def ringtone_test(self, nfft: int = 256, noverlap: int = -1, visualize_output: bool = False):
        """
        Description: FFT of two signals
        Inputs: 
            fft_size: int -- size of fft to perform
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        sample_rate   = 4000
        length_ts_sec = 3
        ## --------------------------------- ##
        ## 3 seconds of "digit 1" sound
        ## Pressing digit 2 buttom generates 
        ## the sine waves at frequency 
        ## 697Hz and 1209Hz.
        ## --------------------------------- ##
        ts1  = np.array(self.get_signal_Hz(697, sample_rate,length_ts_sec)) 
        ts1 += np.array(self.get_signal_Hz(1209,sample_rate,length_ts_sec))
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
        ts2  = np.array(self.get_signal_Hz(697, sample_rate,length_ts_sec)) 
        ts2 += np.array(self.get_signal_Hz(1336,sample_rate,length_ts_sec))
        ts2  = list(ts2)

        ## -------------------- ##
        ## Add up to 7 seconds
        ## ------------------- ##
        # concatenates the lists, doesn't sum their values like a numpy array
        ts = ts1 + ts_silence  + ts2

        print("About to take cuSTFT")
        result = cu.cuSTFT(list(ts), nfft, noverlap)
        for i in range(result.shape[1]):
            result[:, i] = 5 * np.log10(np.abs(result[:, i])**2 / (nfft*sample_rate), where=np.abs(result[:,i])>0)
        f_copy = np.linspace(0, sample_rate, result.shape[0])
        t_copy = np.linspace(0, len(ts)/sample_rate, result.shape[1])
        
        print("Done cuSTFT")
        print("Type: {}".format(type(result)))
        print()
        # result_f = np.linspace(0, )
        f, t, stft_results = scipy.signal.stft(ts, fs=sample_rate, window="boxcar", nperseg=256, noverlap=None if noverlap == -1 else noverlap, nfft=nfft, detrend=False, return_onesided=False, boundary=None, padded=False, axis=-1, scaling='psd')
        for i in range(stft_results.shape[1]):
            stft_results[:, i] = 10*np.log10(np.abs(stft_results[:, i]), where=np.abs(stft_results[:,i])>0)
        # print(f)
        # print('\n')
        # print(t)
        # print('\n')
        # print(t_copy)
        # print('\n')
        # fft_result = np.fft.fft(ts[:nfft])
        # windowed_fft = (2 * fft_result) / (sample_rate*(nfft//2))
        # windowed_fft = 5 * np.log10(np.abs(fft_result)**2 / ((nfft*sample_rate)), where=fft_result > 0)
        for i in range(217):
            print("{}".format(all(np.isclose(result[:,i], stft_results[:,i]))))

        # print("isclose: {}".format(np.isclose(result[:, 0], windowed_fft)))
        # print(result[:10, 0])
        # print('\n')
        # print(fft_result[:10])
        # print('\n')
        # print(windowed_fft[:10])
        # print('\n')
        # print(10*np.log10(np.abs(stft_results[:10,0])))
        # print('\n')
        # print(stft_results[:10, 0])
        # print('\n')

        # tb_result = np.allclose(result, correct_result)

        if visualize_output:
            print("cu_stft.shape(): {}, signal.stft.shape: {}".format(result.shape, stft_results.shape))
            plt.title("Scipy STFT")
            plt.pcolormesh(t, f, np.abs(stft_results))
            plt.savefig(dir_path + '/MatplotGraphs/stft_results.png')

            plt.title("cu_STFT")
            # plt.pcolormesh(t_copy, f, 10.0*np.log10(np.abs(result)))
            plt.pcolormesh(t_copy, f, np.abs(result))
            # plt.pcolormesh(t_copy, f, result)
            plt.savefig(dir_path + '/MatplotGraphs/cuSTFT_results.png')
            # print("Close: {}".format(tb_result))
            # for i in range(len(correct_result)):
            #     print("correct: {}, your output: {}".format(correct_result[i], result[i]))

        # self.tb_results["ringtone_test"] = tb_result

if __name__ == "__main__":
    tb = testbench()

    # running tests
    # tb.simple_one()
    tb.ringtone_test(visualize_output=True)

    # print results
    print(tb.tb_results)