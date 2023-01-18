#include "dsp/dsp.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace dsp;

namespace py = pybind11;

int test_function() {return 1;}

int py_thread_per_block() { return get_thread_per_block();}



PYBIND11_MODULE(dsp_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    // module_handle.def("cuFFT", &pybind_cuFFT);
    module_handle.def("get_threads", &py_thread_per_block);
    module_handle.def("test_func", &test_function);
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