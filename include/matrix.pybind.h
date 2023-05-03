#ifndef _MATRIX_PYBIND_H_
#define _MATRIX_PYBIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "dsp.h"
#include <iostream>
#include <vector>
#include <utility>

using namespace std;

namespace py = pybind11;

/* try to invoke base class deconstructor later */

class py_Matrix {
    public:
    py_Matrix();
    explicit py_Matrix(int rows, int cols);
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

class py_Matrix_float {
    public:
    py_Matrix_float();
    explicit py_Matrix_float(int rows, int cols);
    py_Matrix_float(int rows, int cols, float* data);
    py_Matrix_float(const py_Matrix_float &s);
    py_Matrix_float(py_Matrix_float &&s) noexcept;
    ~py_Matrix_float();
    py_Matrix_float &operator=(const py_Matrix_float &s);
    py_Matrix_float &operator=(py_Matrix_float &&s) noexcept;
    float operator()(int i, int j) const;
    float &operator()(int i, int j);
    float *data();
    py::ssize_t rows() const;
    py::ssize_t cols() const;
    py::tuple shape() const;

    protected:
    py::size_t m_rows;
    py::size_t m_cols;
    float *m_data;

};

class py_stft_Matrix : public py_Matrix {
    public:
    py_stft_Matrix(int rows, int cols) : py_Matrix::py_Matrix(rows, cols) {};
    py_stft_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided, int window, bool mag);
};

class py_mfcc_Matrix : public py_Matrix {
    public:
    py_mfcc_Matrix(int rows, int cols) : py_Matrix::py_Matrix(rows, cols) {};
    py_mfcc_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq);
    py_mfcc_Matrix(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps);
};

class py_mfcc_Matrix_float : public py_Matrix_float {
    public:
    py_mfcc_Matrix_float(int rows, int cols) : py_Matrix_float::py_Matrix_float(rows, cols) {};
    py_mfcc_Matrix_float(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq);
    py_mfcc_Matrix_float(vector<float> samples, int sample_rate, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps);
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