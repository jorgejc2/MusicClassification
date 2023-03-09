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
# import dsp_module as cu
# import wav_module as wav
import matrix_module as myMatrix

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


if __name__ == "__main__":
    print("Current file path is " + dir_path + '\n')
    tb = testbench()

    # running tests
    # tb.simple_test_one()
    # tb.simple_test_two()
    # tb.simple_test_three()
    tb.simple_test_four()

    # print results
    print(tb.tb_results)