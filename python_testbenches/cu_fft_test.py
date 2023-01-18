"""
Jorge Chavez

Testbench used to test CUDA code for cuFFT
"""

# set python file's path directory of modules 
import sys
sys.path.append('../build/pybind_modules')
# rest of imports
import numpy as np
import dsp_module as cu

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

    def simple_two(self, visualize_output: bool = False):
        """
        Description: Basic FFT with 128 numbers
        Inputs: 
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        
        # checking math on exponentials 
        FFT_size = 128

        test = np.array([i for i in range(FFT_size)])

        result = cu.cuFFT(list(test))
        correct_result = np.fft.fft(test)

        tb_result = np.allclose(result, correct_result)

        if visualize_output:
            print("Close: {}".format(tb_result))
            for i in range(len(correct_result)):
                print("correct: {}, your output: {}".format(correct_result[i], result[i]))

        self.tb_results["simple_two"] = tb_result
    
    def simple_three(self, visualize_output: bool = False):
        """
        Description: Basic FFT with 1024 numbers
        Inputs: 
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        
        # checking math on exponentials 
        FFT_size = 1024

        test = np.array([i for i in range(FFT_size)])

        result = cu.cuFFT(list(test))
        correct_result = np.fft.fft(test)

        tb_result = np.allclose(result, correct_result)

        if visualize_output:
            print("Close: {}".format(tb_result))
            for i in range(len(correct_result)):
                print("correct: {}, your output: {}".format(correct_result[i], result[i]))

        self.tb_results["simple_three"] = tb_result

    def ringtone_test(self, fft_size: int = 1024, visualize_output: bool = False):
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

        # test_cuda()

        result = cu.cuFFT(list(ts[:fft_size]))
        correct_result = np.fft.fft(ts[:fft_size])

        tb_result = np.allclose(result, correct_result)

        if visualize_output:
            print("Close: {}".format(tb_result))
            for i in range(len(correct_result)):
                print("correct: {}, your output: {}".format(correct_result[i], result[i]))

        self.tb_results["ringtone_test"] = tb_result

if __name__ == "__main__":
    tb = testbench()

    # running tests
    tb.simple_one()
    tb.simple_two()
    tb.simple_three()
    tb.ringtone_test()

    # print results
    print(tb.tb_results)