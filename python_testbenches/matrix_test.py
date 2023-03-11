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
from time import time

import matrix_module as myMatrix
import wav_module as wav

dir_path = os.path.dirname(os.path.realpath(__file__))

class testbench():
    
    # testbench results

    tb_results = {
        "simple_test_one" : False
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
    def simple_test_one(self):
        # create a simple matrix and cast to numpy without allocating additional memory
        test_matrix = myMatrix.Matrix(2,2)
        print("test matrix: {}".format(test_matrix))
        print("rows: {}".format(test_matrix.rows()))
        print("cols: {}".format(test_matrix.cols()))
        print("data type: {}".format(type(test_matrix)))
        print("data shape: {}".format(test_matrix.shape))
        print("printing data in matrix")
        for i in range(test_matrix.rows()):
            for j in range(test_matrix.cols()):
                print(test_matrix[i,j], end=' | ')
            print('') # print a newline

    def simple_test_two(self):
        # create a simple matrix and cast to numpy without allocating additional memory
        test_matrix = myMatrix.Matrix(np.array([[0,1],[2,3]]).astype(np.float32))
        print("test matrix: {}".format(test_matrix))
        print("rows: {}".format(test_matrix.rows()))
        print("cols: {}".format(test_matrix.cols()))
        print("data type: {}".format(type(test_matrix)))
        print("data shape: {}".format(test_matrix.shape))
        print("printing data in matrix")
        for i in range(test_matrix.rows()):
            for j in range(test_matrix.cols()):
                print(test_matrix[i,j], end=' | ')
            print('') # print a newline

    def simple_test_three(self):
        # create a simple matrix and cast to numpy without allocating additional memory
        test_matrix = myMatrix.c_return_data()
        print("test matrix: {}".format(test_matrix))
        print("rows: {}".format(test_matrix.rows()))
        print("cols: {}".format(test_matrix.cols()))
        print("data type: {}".format(type(test_matrix)))
        print("data shape: {}".format(test_matrix.shape))
        print("printing data in test_matrix")
        for i in range(test_matrix.rows()):
            for j in range(test_matrix.cols()):
                print(test_matrix[i,j], end=' | ')
            print('') # print a newline
        print('')

        # now convert to numpy
        print("printing data in numpy array of test_matrix")
        np_test_matrix = np.array(test_matrix, copy=False) # setting copy to false so that numpy doesn't make a copy of the object's instance
        for i in range(np_test_matrix.shape[0]):
            for j in range(np_test_matrix.shape[1]):
                print(np_test_matrix[i,j], end=' | ')
            print('') # print a newline
        print('')

        # Now manipulate numpy array with numpy functions
        # NOTE: Manipulations might not break the instance since Numpy is good about just creating copies if trying to do operations such as appending
        np_test_matrix = np.where(np_test_matrix < 5, np_test_matrix, 10*np_test_matrix)
        print("Testing np.where on np_test_matrix")
        print(np_test_matrix)

    def simple_test_four(self):
        # create a simple matrix and cast to numpy without allocating additional memory
        test_matrix = myMatrix.c_return_3d_data()
        print("test matrix: {}".format(test_matrix))
        print("width: {}".format(test_matrix.width()))
        print("rows: {}".format(test_matrix.rows()))
        print("cols: {}".format(test_matrix.cols()))
        print("data type: {}".format(type(test_matrix)))
        print("data shape: {}".format(test_matrix.shape))
        print("printing data in test_matrix")
        for z in range(test_matrix.width()):
            print("row: {}".format(z))
            for i in range(test_matrix.rows()):
                for j in range(test_matrix.cols()):
                    print(test_matrix[z,i,j], end=' | ')
                print('') # print a newline
        print('')

        for z in range(test_matrix.width()):
            for i in range(test_matrix.rows()):
                for j in range(test_matrix.cols()):
                    test_matrix[z,i,j] = z

        for z in range(test_matrix.width()):
            print("row: {}".format(z))
            for i in range(test_matrix.rows()):
                for j in range(test_matrix.cols()):
                    print(test_matrix[z,i,j], end=' | ')
                print('') # print a newline
        print('')

    def gztan_test_with_matrix(self, gztan_file: str = "../GTZAN/classical/classical.00000.wav", nfft: int = 256, noverlap: int = -1, percent_error_threshold: int = 0.01, visualize_output: bool = False):

        # read in a wav file from the GZTAN dataset
        wav_path = gztan_file

        # use personal wav function reader
        wav_out = wav.wavsamples(wav_path)
        # unpack results
        ts = wav_out[1]
        sample_rate = wav_out[0]
        
        # ts = [0 for i in range (10)]
        cu_start = time()
        result = myMatrix.DSP_Matrix(list(ts), sample_rate, nfft, noverlap, True, 0, True)
        cu_end = time()
        result = np.array(result, copy=False)
        print("cuSTFT took {} ms to finish".format((cu_end - cu_start)*1000))
        print("shape of output: {}".format(result.shape))
        print("dtype: {}".format(result.dtype))
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
                # if visualize_output and display_limit > 0: print("\tNot matching with values of scipy: {}, cu: {}".format(curr_stft, curr_result))

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


if __name__ == "__main__":
    print("Current file path is " + dir_path + '\n')
    tb = testbench()

    # running tests
    # tb.simple_test_one()
    # tb.simple_test_two()
    # tb.simple_test_three()
    print("#### SIMPLE TEST FOUR ####")
    tb.simple_test_four()
    print("#### GZTAN TEST ####")
    tb.gztan_test_with_matrix(visualize_output=True)

    # print results
    # print(tb.tb_results)