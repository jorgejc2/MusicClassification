#include "matrix.pybind.h"

/****************/
/*
    Original py_Matrix
*/
/****************/
py_Matrix::py_Matrix() {
    m_rows = 0;
    m_cols = 0;
    m_data = nullptr;
}
/* this is a constructor meant to be called from a c++ function, not from Python */
py_Matrix::py_Matrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new double[(size_t) (rows * cols)];
    m_data = (double*)malloc(rows*cols*sizeof(double));
    memset(m_data, 0, sizeof(double) * (size_t) (rows * cols));
}

py_Matrix::py_Matrix(int rows, int cols, double* data) : m_rows(rows), m_cols(cols) {
    m_data = data;
}

py_Matrix::py_Matrix(const py_Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new double[(size_t) (m_rows * m_cols)];
    m_data = (double*)malloc(m_rows*m_cols*sizeof(double));
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
    // delete[] m_data;
    free(m_data);
}

py_Matrix& py_Matrix::operator=(const py_Matrix &s) {
    if (this == &s) {
        return *this;
    }
    // print_copy_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // delete[] m_data;
    free(m_data);
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    // m_data = new double[(size_t) (m_rows * m_cols)];
    m_data = (double*)malloc(m_rows*m_cols*sizeof(double));
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_rows * m_cols));
    return *this;
}

py_Matrix& py_Matrix::operator=(py_Matrix &&s) noexcept {
    // print_move_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    if (&s != this) {
        // delete[] m_data;
        free(m_data);
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

/****************/
/*
    float type of py_Matrix
*/
/****************/
py_Matrix_float::py_Matrix_float() {
    m_rows = 0;
    m_cols = 0;
    m_data = nullptr;
}
/* this is a constructor meant to be called from a c++ function, not from Python */
py_Matrix_float::py_Matrix_float(int rows, int cols) : m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new double[(size_t) (rows * cols)];
    m_data = (float*)malloc(rows*cols*sizeof(float));
    memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
}

py_Matrix_float::py_Matrix_float(int rows, int cols, float* data) : m_rows(rows), m_cols(cols) {
    m_data = data;
}

py_Matrix_float::py_Matrix_float(const py_Matrix_float &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new float[(size_t) (m_rows * m_cols)];
    m_data = (float*)malloc(m_rows*m_cols*sizeof(float));
    memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
}

py_Matrix_float::py_Matrix_float(py_Matrix_float &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
    // print_move_created(this);
    s.m_rows = 0;
    s.m_cols = 0;
    s.m_data = nullptr;
}

py_Matrix_float::~py_Matrix_float() {
    // print_destroyed(this,
    //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // delete[] m_data;
    free(m_data);
}

py_Matrix_float& py_Matrix_float::operator=(const py_Matrix_float &s) {
    if (this == &s) {
        return *this;
    }

    free(m_data);
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    m_data = (float*)malloc(m_rows*m_cols*sizeof(float));
    memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
    return *this;
}

py_Matrix_float& py_Matrix_float::operator=(py_Matrix_float &&s) noexcept {

    if (&s != this) {
        free(m_data);
        m_rows = s.m_rows;
        m_cols = s.m_cols;
        m_data = s.m_data;
        s.m_rows = 0;
        s.m_cols = 0;
        s.m_data = nullptr;
    }
    return *this;
}

float py_Matrix_float::operator()(int i, int j) const {
    return m_data[(size_t) (i * m_cols + j)];
}

float &py_Matrix_float::operator()(int i, int j) {
    return m_data[(size_t) (i * m_cols + j)];
}

float *py_Matrix_float::data() { return m_data; }

py::ssize_t py_Matrix_float::rows() const { return m_rows; }
py::ssize_t py_Matrix_float::cols() const { return m_cols; }
py::tuple py_Matrix_float::shape() const { return py::make_tuple(m_rows, m_cols); }

/****************/
/*
    Definition of py_stft_Matrix class that extends py_Matrix
*/
/****************/
py_stft_Matrix::py_stft_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided, int window, bool mag) {
    pair<int,int> init_dimensions;
    double** set_m_data = &m_data;
    dsp::cuSTFT_vector_in(samples, set_m_data, sample_rate, NFFT, init_dimensions, noverlap, one_sided, window, mag);
    m_rows = init_dimensions.first;
    m_cols = init_dimensions.second;
}

/****************/
/*
    Definition of py_mfcc_Matrix class that extends py_Matrix
*/
/****************/
py_mfcc_Matrix::py_mfcc_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq) {
    pair<int,int> init_dimensions;
    double** set_m_data = &m_data;
    dsp::cuMFCC_vector_in(samples, set_m_data, sample_rate, NFFT, init_dimensions, noverlap, window, preemphasis_b, nfilt, num_ceps, hz_high_freq);
    m_rows = init_dimensions.first;
    m_cols = init_dimensions.second;
    // cout<<"rows: "<<m_rows<<endl;
    // cout<<"cols: "<<m_cols<<endl;
}

py_mfcc_Matrix::py_mfcc_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps) {
    pair<int,int> init_dimensions;
    double** set_m_data = &m_data;
    dsp::cuMFCC_vector_in(samples, set_m_data, sample_rate, NFFT, init_dimensions, noverlap, window, preemphasis_b, nfilt, num_ceps, sample_rate/2);
    m_rows = init_dimensions.first;
    m_cols = init_dimensions.second;
    // cout<<"rows: "<<m_rows<<endl;
    // cout<<"cols: "<<m_cols<<endl;
}


/****************/
/*
    Definition of py_mfcc_Matrix_float class that extends py_Matrix_float
*/
/****************/
py_mfcc_Matrix_float::py_mfcc_Matrix_float(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq) {
    pair<int,int> init_dimensions;
    float** set_m_data = &m_data;
    dsp::cuMFCC_vector_in_float(samples, set_m_data, sample_rate, NFFT, init_dimensions, noverlap, window, preemphasis_b, nfilt, num_ceps, hz_high_freq);
    m_rows = init_dimensions.first;
    m_cols = init_dimensions.second;
}

py_mfcc_Matrix_float::py_mfcc_Matrix_float(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps) {
    pair<int,int> init_dimensions;
    float** set_m_data = &m_data;
    dsp::cuMFCC_vector_in_float(samples, set_m_data, sample_rate, NFFT, init_dimensions, noverlap, window, preemphasis_b, nfilt, num_ceps, sample_rate/2);
    m_rows = init_dimensions.first;
    m_cols = init_dimensions.second;

}

/* this is a constructor meant to be called from a c++ function, not from Python */
py_Matrix3d::py_Matrix3d(int width, int rows, int cols) : m_width(width), m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new double[(size_t) (width * rows * cols)];
    m_data = (double*)malloc(width*rows*cols*sizeof(double));
    memset(m_data, 0, sizeof(double) * (size_t) (width * rows * cols));
}

py_Matrix3d::py_Matrix3d(int width, int rows, int cols, double* data) : m_width(width), m_rows(rows), m_cols(cols) {
    m_data = data;
}

py_Matrix3d::py_Matrix3d(const py_Matrix3d &s) : m_width(s.m_width), m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    // m_data = new double[(size_t) (m_width * m_rows * m_cols)];
    m_data = (double*)malloc(m_width*m_rows*m_cols*sizeof(double));
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
    // delete[] m_data;
    free(m_data);
}

py_Matrix3d& py_Matrix3d::operator=(const py_Matrix3d &s) {
    if (this == &s) {
        return *this;
    }
    // print_copy_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // delete[] m_data;
    free(m_data);
    m_width = s.m_width;
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    // m_data = new double[(size_t) (m_width * m_rows * m_cols)];
    m_data = (double*)malloc(m_width*m_rows*m_cols*sizeof(double));
    memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_width * m_rows * m_cols));
    return *this;
}

py_Matrix3d& py_Matrix3d::operator=(py_Matrix3d &&s) noexcept {
    // print_move_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    if (&s != this) {
        // delete[] m_data;
        free(m_data);
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

    py::class_<py_Matrix_float>(module_handle, "Matrix_float", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_Matrix_float(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &py_Matrix_float::rows)
        .def("cols", &py_Matrix_float::cols)
        .def("shape", &py_Matrix_float::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_Matrix_float &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](py_Matrix_float &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_Matrix_float &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(float) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(float)});
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
    
    py::class_<py_mfcc_Matrix>(module_handle, "MFCC_Matrix", py::buffer_protocol())
        // .def(py::init<py::ssize_t, py::ssize_t>())
        .def(py::init<vector<float>, int, int, int, int, float, int, int, float>())
        .def(py::init<vector<float>, int, int, int, int, float, int, int>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_mfcc_Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &py_mfcc_Matrix::rows)
        .def("cols", &py_mfcc_Matrix::cols)
        .def("shape", &py_mfcc_Matrix::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_mfcc_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](py_mfcc_Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, double v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_mfcc_Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(double)});
        });

        py::class_<py_mfcc_Matrix_float>(module_handle, "MFCC_Matrix_float", py::buffer_protocol())
        // .def(py::init<py::ssize_t, py::ssize_t>())
        .def(py::init<vector<float>, int, int, int, int, float, int, int, float>())
        .def(py::init<vector<float>, int, int, int, int, float, int, int>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new py_mfcc_Matrix_float(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &py_mfcc_Matrix_float::rows)
        .def("cols", &py_mfcc_Matrix_float::cols)
        .def("shape", &py_mfcc_Matrix_float::shape)

        /// Bare bones interface
        .def("__getitem__",
            [](const py_mfcc_Matrix_float &m, std::pair<py::ssize_t, py::ssize_t> i) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                return m(i.first, i.second);
            })
        .def("__setitem__",
            [](py_mfcc_Matrix_float &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                if (i.first >= m.rows() || i.second >= m.cols()) {
                    throw py::index_error();
                }
                m(i.first, i.second) = v;
            })
        /// Provide buffer access
        .def_buffer([](py_mfcc_Matrix_float &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(float) * size_t(m.cols()), /* Strides (in bytes) for each index */
                sizeof(float)});
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