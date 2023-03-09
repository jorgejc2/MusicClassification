#include "matrix_pybind/matrix_pybind.h"

/* this is a constructor meant to be called from a c++ function, not from Python */
py_Matrix::py_Matrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new double[(size_t) (rows * cols)];
    memset(m_data, 0, sizeof(double) * (size_t) (rows * cols));
}

py_Matrix::py_Matrix(int rows, int cols, double* data) : m_rows(rows), m_cols(cols) {
    m_data = data;
}

py_Matrix::py_Matrix(const py_Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new double[(size_t) (m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_rows * m_cols));
}

py_Matrix::py_Matrix(py_Matrix &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
    // print_move_created(this);
    s.m_rows = 0;
    s.m_cols = 0;
    s.m_data = nullptr;
}

py_Matrix::~py_Matrix() {
    // print_destroyed(this,
    //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
}

py_Matrix& py_Matrix::operator=(const py_Matrix &s) {
    if (this == &s) {
        return *this;
    }
    // print_copy_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    m_data = new double[(size_t) (m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_rows * m_cols));
    return *this;
}

py_Matrix& py_Matrix::operator=(py_Matrix &&s) noexcept {
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

double py_Matrix::operator()(int i, int j) const {
    return m_data[(size_t) (i * m_cols + j)];
}

double &py_Matrix::operator()(int i, int j) {
    return m_data[(size_t) (i * m_cols + j)];
}

double *py_Matrix::data() { return m_data; }

py::ssize_t py_Matrix::rows() const { return m_rows; }
py::ssize_t py_Matrix::cols() const { return m_cols; }
py::tuple py_Matrix::shape() const { return py::make_tuple(m_rows, m_cols); }


/* this is a constructor meant to be called from a c++ function, not from Python */
py_Matrix3d::py_Matrix3d(int width, int rows, int cols) : m_width(width), m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new double[(size_t) (width * rows * cols)];
    memset(m_data, 0, sizeof(double) * (size_t) (width * rows * cols));
}

py_Matrix3d::py_Matrix3d(int width, int rows, int cols, double* data) : m_width(width), m_rows(rows), m_cols(cols) {
    m_data = data;
}

py_Matrix3d::py_Matrix3d(const py_Matrix3d &s) : m_width(s.m_width), m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new double[(size_t) (m_width * m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_width * m_rows * m_cols));
}

py_Matrix3d::py_Matrix3d(py_Matrix3d &&s) noexcept : m_width(s.m_width), m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
    // print_move_created(this);
    s.m_width = 0;
    s.m_rows = 0;
    s.m_cols = 0;
    s.m_data = nullptr;
}

py_Matrix3d::~py_Matrix3d() {
    // print_destroyed(this,
    //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
}

py_Matrix3d& py_Matrix3d::operator=(const py_Matrix3d &s) {
    if (this == &s) {
        return *this;
    }
    // print_copy_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
    m_width = s.m_width;
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    m_data = new double[(size_t) (m_width * m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_width * m_rows * m_cols));
    return *this;
}

py_Matrix3d& py_Matrix3d::operator=(py_Matrix3d &&s) noexcept {
    // print_move_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    if (&s != this) {
        delete[] m_data;
        m_width = s.m_width;
        m_rows = s.m_rows;
        m_cols = s.m_cols;
        m_data = s.m_data;
        s.m_width = 0;
        s.m_rows = 0;
        s.m_cols = 0;
        s.m_data = nullptr;
    }
    return *this;
}

double py_Matrix3d::operator()(int z, int i, int j) const {
    return m_data[(size_t) (z*m_cols*m_rows + i * m_cols + j)];
}

double &py_Matrix3d::operator()(int z, int i, int j) {
    return m_data[(size_t) (z*m_cols*m_rows + i * m_cols + j)];
}

double *py_Matrix3d::data() { return m_data; }

py::ssize_t py_Matrix3d::rows() const { return m_rows; }
py::ssize_t py_Matrix3d::cols() const { return m_cols; }
py::ssize_t py_Matrix3d::width() const { return m_width; }
py::tuple py_Matrix3d::shape() const { return py::make_tuple(m_width, m_rows, m_cols); }

py_Matrix c_created_data() {
    int rows = 10;
    int cols = 10;

    py_Matrix temp(10,10);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp(i,j) = i;
        }
    }
    return temp;
}

py_Matrix3d c_created_3d_data() {
    int rows = 10;
    int cols = 10;
    int width = 3;

    py_Matrix3d temp(width, rows, cols);
    for (int z = 0; z < width; z++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                temp(z,i,j) = i;
            }
        }
    }
    return temp;
}

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