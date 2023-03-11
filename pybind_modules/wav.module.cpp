#include "wav.pybind.h"

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