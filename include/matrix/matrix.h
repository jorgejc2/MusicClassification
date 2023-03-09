#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <vector>
#include <string.h>

using namespace std;

class Matrix {
    public:
    Matrix(int rows, int cols);
    Matrix(const Matrix &s);
    Matrix(Matrix &&s) noexcept;
    ~Matrix();
    Matrix &operator=(const Matrix &s);
    Matrix &operator=(Matrix &&s) noexcept;
    float operator()(int i, int j) const;
    float &operator()(int i, int j);
    float *data();
    size_t rows() const;
    size_t cols() const;

    protected:
    size_t m_rows;
    size_t m_cols;
    float *m_data;

};

class Matrix3d {
    public:
    Matrix3d(int width, int rows, int cols);
    Matrix3d(const Matrix3d &s);
    Matrix3d(Matrix3d &&s) noexcept;
    ~Matrix3d();
    Matrix3d &operator=(const Matrix3d &s);
    Matrix3d &operator=(Matrix3d &&s) noexcept;
    float operator()(int z, int i, int j) const;
    float &operator()(int z, int i, int j);
    float *data();
    size_t width() const;
    size_t rows() const;
    size_t cols() const;

    protected:
    size_t m_width;
    size_t m_rows;
    size_t m_cols;
    float *m_data;

};


#endif // _MATRIX_H_