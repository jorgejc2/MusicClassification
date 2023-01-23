// #include "dsp_pybind/dsp.pybind.h"
#include "wav_pybind/wav.pybind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

vector<int16_t> pybind_wavread(char* fileIn) {

    // const char* filePath = &fileIn[0];
    const char* filePath = fileIn;

    wavFileReader wav_obj;
    
    int8_t* wav_samples;
    int num_samples = wav_obj.readFile(&wav_samples, filePath, 1);

    if (num_samples < 0) {
        return vector<int16_t>(1, 0);
    }
    
    int16_t* wav_samples_16 = (int16_t *)wav_samples;

    // int num_samples = 1;
    vector<int16_t> ret_samples(num_samples, 0);

    for (int i = 0; i < num_samples / 2; i++)
        ret_samples[i] = wav_samples_16[i];

    delete [] wav_samples;

    return ret_samples;
}

int test_function() { 
    return 1; 
}

// int library_attached() {
//         // cout<< N <<endl; 
//     vector<int> test_vector(N, 1);
//     printf("test_vector size: %d\n", test_vector.size());
//     return 1;
// }

/*
    Description: Module to be imported by a Python file describing how each function should be interpreted
*/
PYBIND11_MODULE(wav_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("wavsamples", [](char* fileName) {
        py::array out = py::cast(pybind_wavread(fileName));
        return out;
    }, py::arg("fileName"), py::return_value_policy::move);
    module_handle.def("test_function", &test_function);
    // module_handle.def("get_thread_per_block", &dsp::get_thread_per_block);
    // module_handle.def("cuFFT", &pybind_cuFFT, py::return_value_policy::copy);
    // module_handle.def("cuSTFT", [](vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided) {
    //     py::array out = py::cast(pybind_cuSTFT(samples, sample_rate, NFFT, noverlap, one_sided));
    //     return out;
    // }, py::arg("samples"), py::arg("sample_rate"), py::arg("NFFT"), py::arg("noverlap"), py::arg("one_sided"), py::return_value_policy::move);
    // module_handle.def("test_cuda", &test_cuda);
}