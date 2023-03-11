#include "matrix.pybind.h"

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
