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
import wav_module as wav

dir_path = os.path.dirname(os.path.realpath(__file__))

class testbench():
    
    # testbench results

    tb_results = {
        "ringtone_test" : False
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

    def ringtone_test(self, nfft: int = 256, noverlap: int = -1, percent_error_threshold: int = 0.01, visualize_output: bool = False, use_defaults=False):
        """
        Description: FFT of two signals
        Inputs: 
            fft_size: int -- size of fft to perform
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        if use_defaults:
            nfft = 1024
            noverlap = -1
            
        sample_rate   = 4000
        length_ts_sec = 4
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
        result = None
        if use_defaults:
            result = cu.cuSTFT(list(ts), sample_rate, nfft, noverlap, True, 0, True)
        else:
            result = cu.cuSTFT(list(ts), sample_rate)
        t_copy = np.linspace(0, len(ts)/sample_rate, result.shape[1])
        
        # calculate stft from scipy's library 
        f, t, stft_results = scipy.signal.stft(ts, fs=sample_rate, window="boxcar", nperseg=nfft, noverlap=None if noverlap == -1 else noverlap, nfft=nfft, detrend=False, return_onesided=True, boundary=None, padded=False, axis=-1, scaling='psd')
        # calculate power spectral density of every entry
        stft_results = np.abs(stft_results)
        stft_results = 20*np.log10(stft_results)

        # scipy typically trims last time frame, so only check time frames that exists in both arrays
        lesser_t_len = stft_results.shape[1] if stft_results.shape[1] < result.shape[1] else result.shape[1]

        # automatically true if every match is close
        correct_output = np.allclose(result[:,:lesser_t_len], stft_results[:,:lesser_t_len])

        # find entries where result does not match with stft_results
        counter = 10
        num_notclose = 0
        for i in range(lesser_t_len):
            # iterate through every time frame
            close = all(np.isclose(result[:,i], stft_results[:,i]))
            if not close:
                if visualize_output and counter > 0: print("Not close at time frame {}".format(i))
                for j in range(stft_results.shape[0]):
                    # iterate through every frequency in a time frame
                    curr_stft = stft_results[j,i]
                    curr_result = result[j,i]
                    if not np.isclose(curr_stft, curr_result):
                        # two entries were found not matching
                        num_notclose += 1
                        counter -= 1
                        if visualize_output and counter > 0: print("\tNot matching at row: {}, column: {}; values of scipy: {}, cu: {}".format(i, j, curr_stft, curr_result))

        # total compared entries
        total = stft_results.shape[0] * lesser_t_len
        
        # check if error is permissible for testbench
        if num_notclose/total < percent_error_threshold:
            correct_output = True

        tb_result = correct_output

        if visualize_output:
            print("num_notclose: {}, total: {}, error: {}, error_threshold: {}".format(num_notclose, total, num_notclose/total, percent_error_threshold))
            print("cu_stft.shape(): {}, signal.stft.shape: {}".format(result.shape, stft_results.shape))
            plt.title("Scipy STFT")
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.pcolormesh(t, f, (stft_results))
            plt.colorbar(format="%+2.f dB")
            plt.savefig(dir_path + '/MatplotGraphs/stft_results.png')
            plt.show()
            plt.close()

            plt.title("cu_STFT")
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.pcolormesh(t_copy, f, (result))
            plt.colorbar(format="%+2.f dB")
            plt.savefig(dir_path + '/MatplotGraphs/cuSTFT_results.png')
            plt.show()
            plt.close()

        self.tb_results["ringtone_test"] = tb_result

    def test_defaults(self, visualize_output:bool = False):
        self.ringtone_test(use_defaults=True, visualize_output=visualize_output)
        cu.get_device_properties()

    def gztan_test(self, gztan_file: str = "../GTZAN/classical/classical.00000.wav", nfft: int = 256, noverlap: int = -1, percent_error_threshold: int = 0.01, visualize_output: bool = False):

        # read in a wav file from the GZTAN dataset
        wav_path = gztan_file

        # use personal wav function reader
        wav_out = wav.wavsamples(wav_path)
        # unpack results
        ts = wav_out[1]
        sample_rate = wav_out[0]

        result = cu.cuSTFT(list(ts), sample_rate, nfft, noverlap, True, 0)
        t_copy = np.linspace(0, len(ts)/sample_rate, result.shape[1])
        
        # calculate stft from scipy's library 
        f, t, stft_results = scipy.signal.stft(ts, fs=sample_rate, window="boxcar", nperseg=nfft, noverlap=None if noverlap == -1 else noverlap, nfft=nfft, detrend=False, return_onesided=True, boundary=None, padded=False, axis=-1, scaling='psd')
        # calculate power spectral density of every entry
        stft_results = np.abs(stft_results)
        stft_results = 20*np.log10(stft_results)

        # scipy typically trims last time frame, so only check time frames that exists in both arrays
        lesser_t_len = stft_results.shape[1] if stft_results.shape[1] < result.shape[1] else result.shape[1]

        # automatically true if every match is close
        correct_output = np.allclose(result[:,:lesser_t_len], stft_results[:,:lesser_t_len])

        # find entries where result does not match with stft_results
        num_notclose = 0
        result_1D = result.ravel()
        stft_results_1D = stft_results.ravel()
        display_limit = 20
        for i in range(lesser_t_len):
            curr_stft = stft_results_1D[i]
            curr_result = result_1D[i]
            if not np.isclose(curr_stft, curr_result, rtol=1e-04):
                # two entries were found not matching
                num_notclose += 1
                display_limit -= 1
                if visualize_output and display_limit > 0: print("\tNot matching with values of scipy: {}, cu: {}".format(curr_stft, curr_result))

        # total compared entries
        total = stft_results.shape[0] * lesser_t_len
        
        # check if error is permissible for testbench
        if num_notclose/total < percent_error_threshold:
            correct_output = True

        tb_result = correct_output

        if visualize_output:
            print("num_notclose: {}, total: {}, error: {}, error_threshold: {}".format(num_notclose, total, num_notclose/total, percent_error_threshold))
            print("cu_stft.shape(): {}, signal.stft.shape: {}".format(result.shape, stft_results.shape))
            plt.title("Scipy STFT")
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.pcolormesh(t, f, (stft_results))
            plt.colorbar(format="%+2.f dB")
            plt.savefig(dir_path + '/MatplotGraphs/stft_results_GTZAN.png')
            plt.show()
            plt.close()

            plt.title("cu_STFT")
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.pcolormesh(t_copy, f, (result))
            plt.colorbar(format="%+2.f dB")
            plt.savefig(dir_path + '/MatplotGraphs/cuSTFT_results_GTZAN.png')
            plt.show()
            plt.close()

        self.tb_results["GTZAN_test"] = tb_result

    # def gztan_test_with_matrix(self, gztan_file: str = "../GTZAN/classical/classical.00000.wav", nfft: int = 256, noverlap: int = -1, percent_error_threshold: int = 0.01, visualize_output: bool = False):

    #     # read in a wav file from the GZTAN dataset
    #     wav_path = gztan_file

    #     # use personal wav function reader
    #     wav_out = wav.wavsamples(wav_path)
    #     # unpack results
    #     ts = wav_out[1]
    #     sample_rate = wav_out[0]

    #     result = np.array(cu.cuSTFT(list(ts), sample_rate, nfft, noverlap, True, 0), copy=False)
    #     t_copy = np.linspace(0, len(ts)/sample_rate, result.shape[1])
        
    #     # calculate stft from scipy's library 
    #     f, t, stft_results = scipy.signal.stft(ts, fs=sample_rate, window="boxcar", nperseg=nfft, noverlap=None if noverlap == -1 else noverlap, nfft=nfft, detrend=False, return_onesided=True, boundary=None, padded=False, axis=-1, scaling='psd')
    #     # calculate power spectral density of every entry
    #     stft_results = np.abs(stft_results)
    #     stft_results = 20*np.log10(stft_results)

    #     # scipy typically trims last time frame, so only check time frames that exists in both arrays
    #     lesser_t_len = stft_results.shape[1] if stft_results.shape[1] < result.shape[1] else result.shape[1]

    #     # automatically true if every match is close
    #     correct_output = np.allclose(result[:,:lesser_t_len], stft_results[:,:lesser_t_len])

    #     # find entries where result does not match with stft_results
    #     num_notclose = 0
    #     result_1D = result.ravel()
    #     stft_results_1D = stft_results.ravel()
    #     display_limit = 20
    #     for i in range(lesser_t_len):
    #         curr_stft = stft_results_1D[i]
    #         curr_result = result_1D[i]
    #         if not np.isclose(curr_stft, curr_result, rtol=1e-04):
    #             # two entries were found not matching
    #             num_notclose += 1
    #             display_limit -= 1
    #             if visualize_output and display_limit > 0: print("\tNot matching with values of scipy: {}, cu: {}".format(curr_stft, curr_result))

    #     # total compared entries
    #     total = stft_results.shape[0] * lesser_t_len
        
    #     # check if error is permissible for testbench
    #     if num_notclose/total < percent_error_threshold:
    #         correct_output = True

    #     tb_result = correct_output

    #     if visualize_output:
    #         print("num_notclose: {}, total: {}, error: {}, error_threshold: {}".format(num_notclose, total, num_notclose/total, percent_error_threshold))
    #         print("cu_stft.shape(): {}, signal.stft.shape: {}".format(result.shape, stft_results.shape))
    #         plt.title("Scipy STFT")
    #         plt.xlabel("Time (sec)")
    #         plt.ylabel("Frequency (Hz)")
    #         plt.pcolormesh(t, f, (stft_results))
    #         plt.colorbar(format="%+2.f dB")
    #         plt.savefig(dir_path + '/MatplotGraphs/stft_results_GTZAN.png')
    #         plt.show()
    #         plt.close()

    #         plt.title("cu_STFT")
    #         plt.xlabel("Time (sec)")
    #         plt.ylabel("Frequency (Hz)")
    #         plt.pcolormesh(t_copy, f, (result))
    #         plt.colorbar(format="%+2.f dB")
    #         plt.savefig(dir_path + '/MatplotGraphs/cuSTFT_results_GTZAN.png')
    #         plt.show()
    #         plt.close()

    #     self.tb_results["GTZAN_test"] = tb_result

if __name__ == "__main__":
    print("Current file path is " + dir_path + '\n')
    tb = testbench()

    # running tests
    # tb.simple_one()
    # tb.ringtone_test(visualize_output=True)
    tb.gztan_test("../GTZAN/pop/pop.00000.wav", visualize_output=True)
    # tb.gztan_test_with_matrix("../GTZAN/pop/pop.00000.wav", visualize_output=True)
    # tb.test_defaults(visualize_output=True)

    # print results
    print(tb.tb_results)