#ifndef _DSP_H_
#define _DSP_H_

#include <NumCpp.hpp>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;
typedef complex<float> dcomp;

const dcomp img(0.0,1.0);

namespace dsp {
    int create_spectogram(vector<float> *ts, int NFFT, int noverlap);
}

#endif // _DSP_H_