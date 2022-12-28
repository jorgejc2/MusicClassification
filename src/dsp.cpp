#include "dsp/dsp.h"
#include <iostream>

int dsp::create_spectogram(vector<float> *ts, int NFFT = 256, int noverlap = -1) {
    if (noverlap < 0)
        noverlap = NFFT / 2;

    nc::NdArray<int> starts_original = nc::arange<int>(0, ts->size(), NFFT - noverlap);
    nc::NdArray<int> starts = starts_original[starts_original + NFFT < (int)ts->size()];
    /* create a 2D vector where rows represent each time window and columns represent frequency bins */
    vector<vector<float>> xns (starts.size(), vector<float>(ts->size()/2));
    bool p_flag = false;

    nc::NdArray<int> ks = nc::arange<int>(0, NFFT, 1);
    printf("%ld computations will occur\n", starts.size() * (ts->size()/2) * NFFT);
    for (int m = 0; m < starts.size(); m++) {
        for (int n = 0; n < ts->size()/2; n++) {
            dcomp a = 0;
            float calc = 0.0;
            int ts_offset = starts[m];

            dcomp curr_n = n;
            dcomp curr_NFFT = NFFT;
            dcomp curr_pi = M_PI;
            dcomp two = 2;

            for (int k = 0; k < NFFT; k++) {
                dcomp curr_ts = (*ts)[ts_offset + k];
                dcomp curr_ks = ks[k];
                a += curr_ts * exp((img * two * curr_pi * curr_ks * curr_n)/curr_NFFT);
            }

            // calc = real(a);
            calc = abs(a);

            // if (p_flag == false) {
            //     p_flag = true;
            //     cout<< "a: " << a << "calc: " << calc << " ";
            // }

            xns[m][n] = calc;
        }
    }

    return 0;
}