#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// test_from_python / test_to_python:
class Matrix {
public:
    /* this is a constructor meant to be called from a c++ function, not from Python */
    Matrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
        // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        m_data = new float[(size_t) (rows * cols)];
        memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
    }

    Matrix(py::ssize_t rows, py::ssize_t cols) : m_rows(rows), m_cols(cols) {
        // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        m_data = new float[(size_t) (rows * cols)];
        memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
    }

    Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
        // print_copy_created(this,
        //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        m_data = new float[(size_t) (m_rows * m_cols)];
        memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
    }

    Matrix(Matrix &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
        // print_move_created(this);
        s.m_rows = 0;
        s.m_cols = 0;
        s.m_data = nullptr;
    }

    ~Matrix() {
        // print_destroyed(this,
        //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        delete[] m_data;
    }

    Matrix &operator=(const Matrix &s) {
        if (this == &s) {
            return *this;
        }
        // print_copy_assigned(this,
        //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        delete[] m_data;
        m_rows = s.m_rows;
        m_cols = s.m_cols;
        m_data = new float[(size_t) (m_rows * m_cols)];
        memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
        return *this;
    }

    Matrix &operator=(Matrix &&s) noexcept {
        // print_move_assigned(this,
        //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        if (&s != this) {
            delete[] m_data;
            m_rows = s.m_rows;
            m_cols = s.m_cols;
            m_data = s.m_data;
            s.m_rows = 0;
            s.m_cols = 0;
            s.m_data = nullptr;
        }
        return *this;
    }

    float operator()(py::ssize_t i, py::ssize_t j) const {
        return m_data[(size_t) (i * m_cols + j)];
    }

    float operator()(int i, int j) const {
        return m_data[(size_t) (i * m_cols + j)];
    }

    float &operator()(py::ssize_t i, py::ssize_t j) {
        return m_data[(size_t) (i * m_cols + j)];
    }

    float &operator()(int i, int j) {
        return m_data[(size_t) (i * m_cols + j)];
    }

    float *data() { return m_data; }

    py::ssize_t rows() const { return m_rows; }
    py::ssize_t cols() const { return m_cols; }
    py::tuple shape() const { return py::make_tuple(m_rows, m_cols); }

private:
    py::ssize_t m_rows;
    py::ssize_t m_cols;
    float *m_data;
};

Matrix c_created_data() {
    int rows = 10;
    int cols = 10;

    Matrix temp(10,10);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp(i,j) = i;
        }
    }
    return temp;
}

PYBIND11_MODULE(matrix_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    py::class_<Matrix>(module_handle, "Matrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("shape", &Matrix::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(float) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(float)});
        });
    module_handle.def("c_return_data", &c_created_data);
}

/*
    Description: Module to be imported by a Python file describing how each function should be interpreted
*/
// PYBIND11_MODULE(matrix_module, module_handle) {
//     module_handle.doc() = "I'm a docstring hehe";
//     py::class_<Matrix>(module_handle, "Matrix", py::buffer_protocol())
//    .def_buffer([](Matrix &m) -> py::buffer_info {
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