#include "matrix.pybind.h"

PYBIND11_MODULE(matrix_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    py::class_<py_Matrix>(module_handle, "Matrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &py_Matrix::rows)
        .def("cols", &py_Matrix::cols)
        .def("shape", &py_Matrix::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](py_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, double v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(double)});
        });

    py::class_<py_stft_Matrix>(module_handle, "DSP_Matrix", py::buffer_protocol())
        // .def(py::init<py::ssize_t, py::ssize_t>())
        .def(py::init<vector<float>, int, int, int, bool, int, bool>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_stft_Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &py_stft_Matrix::rows)
        .def("cols", &py_stft_Matrix::cols)
        .def("shape", &py_stft_Matrix::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_stft_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](py_stft_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, double v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_stft_Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(double)});
        });

    py::class_<py_Matrix3d>(module_handle, "Matrix3d", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 3) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_Matrix3d(info.shape[0], info.shape[1], info.shape[2]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->width() * v->rows() * v->cols()));
            return v;
        }))

        .def("width", &py_Matrix3d::width)
        .def("rows", &py_Matrix3d::rows)
        .def("cols", &py_Matrix3d::cols)
        .def("shape", &py_Matrix3d::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_Matrix3d &m, vector<py::ssize_t> i) {
                if (i[0] >= m.width() || i[1] >= m.rows() || i[2] >= m.cols()) {
                    throw py::index_error();
                }
                return m(i[0], i[1], i[2]);
            })
        .def("__setitem__",
            [](py_Matrix3d &m, vector<py::ssize_t> i, double v) {
                if (i[0] >= m.width() || i[1] >= m.rows() || i[2] >= m.cols()) {
                    throw py::index_error();
                }
                m(i[0], i[1], i[2]) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_Matrix3d &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.width(), m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(double)*size_t(m.rows())*size_t(m.cols()), sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(double)});
        });
    module_handle.def("c_return_data", &c_created_data);
    module_handle.def("c_return_3d_data", &c_created_3d_data);
}

/*
    Description: Module to be imported by a Python file describing how each function should be interpreted
*/
// PYBIND11_MODULE(matrix_module, module_handle) {
//     module_handle.doc() = "I'm a docstring hehe";
//     py::class_<py_Matrix>(module_handle, "py_Matrix", py::buffer_protocol())
//    .def_buffer([](py_Matrix &m) -> py::buffer_info {
//         return py::buffer_info(
//             m.data(),                               /* Pointer to buffer */
//             sizeof(float),                          /* Size of one scalar */
//             py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
//             2,                                      /* Number of dimensions */
//             { m.rows(), m.cols() },                 /* Buffer dimensions */
//             { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
//               sizeof(float) }
//         );
//     });
// /* commented out but kept for reference for adding a class */

// //   module_handle.def("some_fn_python_name", &some_fn);
// //   module_handle.def("some_class_factory", &some_class_factory);
// //   py::class_<SomeClass>(
// // 			module_handle, "PySomeClass"
// // 			).def(py::init<float>())
// //     .def_property("multiplier", &SomeClass::get_mult, &SomeClass::set_mult)
// //     .def("multiply", &SomeClass::multiply)
// //     .def("multiply_list", &SomeClass::multiply_list)
// //     // .def_property_readonly("image", &SomeClass::make_image)
// //     .def_property_readonly("image", [](SomeClass &self) {
// // 				      py::array out = py::cast(self.make_image());
// // 				      return out;
// // 				    })
// //     // .def("multiply_two", &SomeClass::multiply_two)
// //     .def("multiply_two", [](SomeClass &self, float one, float two) {
// // 			   return py::make_tuple(self.multiply(one), self.multiply(two));
// // 			 })
// //     .def("function_that_takes_a_while", &SomeClass::function_that_takes_a_while)
// //     ;
// }