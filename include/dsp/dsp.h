#ifndef _DSP_H_
#define _DSP_H_

#include <NumCpp.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>

using namespace std::chrono;
using namespace std;

typedef complex<float> dcomp;

const dcomp img(0.0,1.0);

namespace dsp {
    int create_spectogram(vector<float> *ts, int NFFT, int noverlap);

    int DFT_slow(vector<float> *ts, nc::NdArray<int> *ks, vector<float> *xns, int ts_offset, int NFFT);

    int FFT(vector<float> *ts, int NFFT, int noverlap);

    __global__ void vector_add(float *out, float *a, float *b, int n);

    __host__ void get_device_properties();

    __host__ int test_cuda();
}

#endif // _DSP_H_