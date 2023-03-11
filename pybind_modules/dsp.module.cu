#include "dsp.pybind.h"

/*
    Description: Module to be imported by a Python file describing how each function should be interpreted
*/
PYBIND11_MODULE(dsp_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("get_thread_per_block", &dsp::get_thread_per_block);
    module_handle.def("get_device_properties", &dsp::get_device_properties);
    module_handle.def("cuFFT", &pybind_cuFFT, py::return_value_policy::copy);
    module_handle.def("cuSTFT", [](vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided, int window, bool mag) {
        py::array out = py::cast(pybind_cuSTFT(samples, sample_rate, NFFT, noverlap, one_sided, window, mag));
        return out;
    }, py::arg("samples"), py::arg("sample_rate"), py::arg_v("NFFT", 1024, "int"), py::arg_v("noverlap", -1, "int"), py::arg_v("one_sided", true, "bool"), py::arg_v("window", 2,"int"), py::arg_v("mag", true, "bool"), py::return_value_policy::move);
    // module_handle.def("cuSTFT_matrix",[](vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided, int window, bool mag) {
    //     return pybind_cuSTFT_matrix(samples, sample_rate, NFFT, noverlap, one_sided, window, mag);
    // }, py::arg("samples"), py::arg("sample_rate"), py::arg_v("NFFT", 1024, "int"), py::arg_v("noverlap", -1, "int"), py::arg_v("one_sided", true, "bool"), py::arg_v("window", 2,"int"), py::arg_v("mag", true, "bool"));
    module_handle.def("test_cuda", &test_cuda);
/* commented out but kept for reference for adding a class */

//   module_handle.def("some_fn_python_name", &some_fn);
//   module_handle.def("some_class_factory", &some_class_factory);
//   py::class_<SomeClass>(
// 			module_handle, "PySomeClass"
// 			).def(py::init<float>())
//     .def_property("multiplier", &SomeClass::get_mult, &SomeClass::set_mult)
//     .def("multiply", &SomeClass::multiply)
//     .def("multiply_list", &SomeClass::multiply_list)
//     // .def_property_readonly("image", &SomeClass::make_image)
//     .def_property_readonly("image", [](SomeClass &self) {
// 				      py::array out = py::cast(self.make_image());
// 				      return out;
// 				    })
//     // .def("multiply_two", &SomeClass::multiply_two)
//     .def("multiply_two", [](SomeClass &self, float one, float two) {
// 			   return py::make_tuple(self.multiply(one), self.multiply(two));
// 			 })
//     .def("function_that_takes_a_while", &SomeClass::function_that_takes_a_while)
//     ;
}