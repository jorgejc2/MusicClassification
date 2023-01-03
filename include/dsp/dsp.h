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
}

#endif // _DSP_H_