#ifndef _DSP_PYBIND_H_
#define _DSP_PYBIND_H_

#include "dsp.h"
#include "matrix.pybind.h"

__host__ int test_cuda();
__host__ vector<complex<double>> pybind_cuFFT(vector<float> samples);
__host__ vector<vector<double>> pybind_cuSTFT(vector<float> samples, int sample_rate, int NFFT, int noverlap, bool one_sided, int window, bool mag);

#endif // _DSP_PYBIND_H_