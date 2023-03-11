#ifndef _WAV_PYBIND_H_
#define _WAV_PYBIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "wavread.h"
#include <iostream>
#include <vector>

namespace py = pybind11;

using namespace std;

// #define N 10000
// __host__ vector<complex<double>> pybind_cuFFT(vector<float> samples);
// vector<int16_t> pybind_wavread(vector<char> fileIn);
vector<int16_t> pybind_wavread(char* fileIn, int* sample_rate);

#endif // _WAV_PYBIND_H_