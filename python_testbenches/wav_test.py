import sys
sys.path.append('../build/pybind_modules')
import numpy as np
from scipy.io import wavfile
import wav_module as wav

class testbench():
    
    # testbench results

    tb_results = {
        "simple_one": False,
        "simple_two": False,
        "simple_three": False
    }
    
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
        wav_path = "../music_samples_wav/chicago.wav"

        # use personal wav function reader
        wav_out = wav.wavsamples(wav_path)
        # unpack results
        wav_samples = wav_out[1]
        wav_fs = wav_out[0]

        # use Scipy's wav file reader
        fs, scipy_samples = wavfile.read(wav_path)

        # check that the resulting arrays are the same
        tb_result = True if wav_samples.all() == scipy_samples.all() else False
        
        if visualize_output:
            print("wav_samples with size {} and type {} and sampling rate {}".format(len(wav_samples), type(wav_samples[0]), wav_fs))
            print("scipy_samples with size {} and type {} and sampling rate {}".format(len(scipy_samples), type(scipy_samples[0]),fs))
            print("Matching: {}".format(tb_result))


        self.tb_results["simple_one"] = tb_result

    def simple_two(self, visualize_output: bool = False):
        """
        Description: Basic FFT with 32 numbers
        Inputs: 
            visualize_output: bool -- print extra information of FFT results
        Ouputs: None
        Returns: None
        Effects: updates tb_results
        """
        
        # checking math on exponentials 
        wav_path = "../GTZAN/classical/classical.00000.wav"

        # use personal wav function reader
        wav_out = wav.wavsamples(wav_path)
        # unpack results
        wav_samples = wav_out[1]
        wav_fs = wav_out[0]

        # use Scipy's wav file reader
        fs, scipy_samples = wavfile.read(wav_path)

        # check that the resulting arrays are the same
        tb_result = True if wav_samples.all() == scipy_samples.all() else False
        
        if visualize_output:
            print("wav_samples with size {} and type {} and sampling rate {}".format(len(wav_samples), type(wav_samples[0]), wav_fs))
            print("scipy_samples with size {} and type {} and sampling rate {}".format(len(scipy_samples), type(scipy_samples[0]),fs))
            print("Matching: {}".format(tb_result))


        self.tb_results["simple_two"] = tb_result


if __name__ == "__main__":
    print("Imports succesful")
    tb = testbench()

    tb.simple_one()
    tb.simple_two()

    print(tb.tb_results)
