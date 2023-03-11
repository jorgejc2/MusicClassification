#include "wav.pybind.h"

/*
    Description: Returns array of samples from a wav file
    Inputs:
        char* fileIn -- file to read
    Outputs:
        int* sample_rate -- sample rate of wav file
    Returns:
        vector<int16_t> -- array of samples
    Effects:
        None
*/
vector<int16_t> pybind_wavread(char* fileIn, int* sample_rate) {

    // const char* filePath = &fileIn[0];
    const char* filePath = fileIn;

    /* create wav reader obejct */
    wavFileReader wav_obj;
    
    /* read wav file */
    int8_t* wav_samples;
    wav_hdr wavHdr = wav_obj.getWavHdr(filePath);
    *sample_rate = wavHdr.SamplesPerSec;
    int num_samples = wav_obj.readFile(&wav_samples, filePath, 1);

    /* return single item array if operation is unsuccessful */
    if (num_samples < 0) {
        return vector<int16_t>(1, 0);
    }

    /* assuming little-endian architecture, place samples into a vector */
    vector<int16_t> ret_samples;
    for (int i = 0; i < num_samples; i=i+2) {
        int16_t curr_sample = wav_samples[i+1] << 8 | wav_samples[i];
        ret_samples.push_back(curr_sample);
    }

    /* free memory */
    delete [] wav_samples;

    /* return sample vector */
    return ret_samples;
}

/*
    Description: Module to be imported by a Python file describing how each function should be interpreted
*/
PYBIND11_MODULE(wav_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("wavsamples", [](char* fileName) {
        int sample_rate;
        py::array_t<int16_t> out = py::cast(pybind_wavread(fileName, &sample_rate));
        py::tuple out_tuple = py::make_tuple(sample_rate, out);
        return out_tuple;
    }, py::arg("fileName"), py::return_value_policy::move);
}