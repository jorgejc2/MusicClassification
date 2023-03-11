#ifndef _MATRIX_PYBIND_H_
#define _MATRIX_PYBIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

using namespace std;

namespace py = pybind11;

/* try to invoke base class deconstructor later */

class py_Matrix {
    public:
    py_Matrix(int rows, int cols);
    py_Matrix(int rows, int cols, double* data);
    py_Matrix(const py_Matrix &s);
    py_Matrix(py_Matrix &&s) noexcept;
    ~py_Matrix();
    py_Matrix &operator=(const py_Matrix &s);
    py_Matrix &operator=(py_Matrix &&s) noexcept;
    double operator()(int i, int j) const;
    double &operator()(int i, int j);
    double *data();
    py::ssize_t rows() const;
    py::ssize_t cols() const;
    py::tuple shape() const;

    protected:
    py::size_t m_rows;
    py::size_t m_cols;
    double *m_data;

};

class py_Matrix3d {
    public:
    py_Matrix3d(int width, int rows, int cols);
    py_Matrix3d(int width, int rows, int cols, double* data);
    py_Matrix3d(const py_Matrix3d &s);
    py_Matrix3d(py_Matrix3d &&s) noexcept;
    ~py_Matrix3d();
    py_Matrix3d &operator=(const py_Matrix3d &s);
    py_Matrix3d &operator=(py_Matrix3d &&s) noexcept;
    double operator()(int z, int i, int j) const;
    double &operator()(int z, int i, int j);
    double *data();
    py::ssize_t width() const;
    py::ssize_t rows() const;
    py::ssize_t cols() const;
    py::tuple shape() const;

    protected:
    py::ssize_t m_width;
    py::ssize_t m_rows;
    py::ssize_t m_cols;
    double *m_data;

};

py_Matrix c_created_data();
py_Matrix3d c_created_3d_data();

#endif // _MATRIX_PYBIND_H_