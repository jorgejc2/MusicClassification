#include "dsp/dsp.h"
#include <iostream>

int dsp::create_spectogram(vector<float> *ts, int NFFT = 256, int noverlap = -1) {
    if (noverlap < 0)
        noverlap = NFFT / 2;

    int32_t ts_size = (int32_t)ts->size();

    nc::NdArray<int> starts_original = nc::arange<int>(0, ts_size, NFFT - noverlap);
    nc::NdArray<int> starts = starts_original[starts_original + NFFT < (int)ts_size];
    /* create a 2D vector where rows represent each time window and columns represent frequency bins */
    vector<vector<float>> xns (starts.size(), vector<float>(ts_size/2));
    bool p_flag = false;

    nc::NdArray<int> ks = nc::arange<int>(0, NFFT, 1);
    printf("%ld computations will occur\n", starts.size() * (ts_size/2) * NFFT);
    auto start = high_resolution_clock::now();
    for (int m = 0; m < starts.size(); m++) {
        for (int n = 0; n < ts_size/2; n++) {
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

            calc = abs(a);

            xns[m][n] = calc;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;
    /* print out the dft values */
    
    for (int m = 0; m < 2; m++) {
        for (int n = 0; n < ts_size / 2; n++) {
            printf("%.3f ", xns[m][n]);
        }
        printf("\n");
    }

    return 0;
}