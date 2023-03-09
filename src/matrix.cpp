#include "matrix/matrix.h"


/* this is a constructor meant to be called from a c++ function, not from Python */
Matrix::Matrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new float[(size_t) (rows * cols)];
    memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
}

Matrix::Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new float[(size_t) (m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
}

Matrix::Matrix(Matrix &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
    // print_move_created(this);
    s.m_rows = 0;
    s.m_cols = 0;
    s.m_data = nullptr;
}

Matrix::~Matrix() {
    // print_destroyed(this,
    //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
}

Matrix& Matrix::operator=(const Matrix &s) {
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

Matrix& Matrix::operator=(Matrix &&s) noexcept {
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

float Matrix::operator()(int i, int j) const {
    return m_data[(size_t) (i * m_cols + j)];
}

float &Matrix::operator()(int i, int j) {
    return m_data[(size_t) (i * m_cols + j)];
}

float *Matrix::data() { return m_data; }

size_t Matrix::rows() const { return m_rows; }
size_t Matrix::cols() const { return m_cols; }


/* this is a constructor meant to be called from a c++ function, not from Python */
Matrix3d::Matrix3d(int width, int rows, int cols) : m_width(width), m_rows(rows), m_cols(cols) {
    // print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new float[(size_t) (width * rows * cols)];
    memset(m_data, 0, sizeof(float) * (size_t) (width * rows * cols));
}

Matrix3d::Matrix3d(const Matrix3d &s) : m_width(s.m_width), m_rows(s.m_rows), m_cols(s.m_cols) {
    // print_copy_created(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_data = new float[(size_t) (m_width * m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_width * m_rows * m_cols));
}

Matrix3d::Matrix3d(Matrix3d &&s) noexcept : m_width(s.m_width), m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
    // print_move_created(this);
    s.m_width = 0;
    s.m_rows = 0;
    s.m_cols = 0;
    s.m_data = nullptr;
}

Matrix3d::~Matrix3d() {
    // print_destroyed(this,
    //                 std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
}

Matrix3d& Matrix3d::operator=(const Matrix3d &s) {
    if (this == &s) {
        return *this;
    }
    // print_copy_assigned(this,
    //                     std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
    delete[] m_data;
    m_width = s.m_width;
    m_rows = s.m_rows;
    m_cols = s.m_cols;
    m_data = new float[(size_t) (m_width * m_rows * m_cols)];
    memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_width * m_rows * m_cols));
    return *this;
}

Matrix3d& Matrix3d::operator=(Matrix3d &&s) noexcept {
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

float Matrix3d::operator()(int z, int i, int j) const {
    return m_data[(size_t) (z*m_cols*m_rows + i * m_cols + j)];
}

float &Matrix3d::operator()(int z, int i, int j) {
    return m_data[(size_t) (z*m_cols*m_rows + i * m_cols + j)];
}

float *Matrix3d::data() { return m_data; }

size_t Matrix3d::rows() const { return m_rows; }
size_t Matrix3d::cols() const { return m_cols; }
size_t Matrix3d::width() const { return m_width; }
