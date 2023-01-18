#ifndef _DSP_PYBIND_H_
#define _DSP_PYBIND_H_

#include "dsp/dsp.h"

__host__ vector<complex<double>> pybind_cuFFT(vector<float> samples);

#endif // _DSP_PYBIND_H_