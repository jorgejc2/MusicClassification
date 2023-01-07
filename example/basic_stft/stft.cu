#include "wavreader/wavread.h"
#include "dsp/dsp.h"
#include <math.h>
#include <vector>
#include <NumCpp.hpp>
using namespace std;

int main(int argc, char* argv[])
{

    int sample_rate = 4000;
    int length_ts_sec = 3;
    int length_ts1_sec = 1;
    int length_ts2_sec = 3;
    int total_ts_length = length_ts_sec + length_ts1_sec + length_ts2_sec;
    int nc_ts_size = sample_rate * 7; // want 7 seconds worth of samples

    int freq1 = 697;
    int freq2 = 1209;
    int freq3 = 1336;

    /* now an attempt to create a linspace vector with NumCpp */
    nc::NdArray<float> lin_freq1 = nc::linspace<float>(0, M_PI*2*freq1, sample_rate);
    nc::NdArray<float> lin_freq2 = nc::linspace<float>(0, M_PI*2*freq2, sample_rate);
    nc::NdArray<float> lin_freq3 = nc::linspace<float>(0, M_PI*2*freq3, sample_rate);

    vector<float>nc_ts(nc_ts_size, 0.0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < sample_rate; j++) {
            nc_ts[i*sample_rate + j] = sin(lin_freq1[j]) + sin(lin_freq2[j]);
            nc_ts[i*sample_rate + j + 4000] = sin(lin_freq1[j]) + sin(lin_freq3[j]);
        }
    }

    for (int i = 0; i < 10; i++)
        printf("%.15f ", nc_ts[i]);
    
    /* if results above match that of Python, then move on to conducting a serial stft */
    // dsp::create_spectogram(&nc_ts, 256, -1);
    // int test_out = dsp::test_cuda();

    int fft_size = 8;

    cuDoubleComplex* freqs = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * fft_size);
    float* cuda_samples = (float*)malloc(sizeof(float) * fft_size);;
    if (freqs == nullptr || cuda_samples == nullptr) {
        printf("failed allocatin \n");
        return 1;
    }

    for (int i = 0; i < fft_size; i++)
        cuda_samples[i] = i;

    dsp::FFT_Setup(cuda_samples, freqs, fft_size);

    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", freqs[i].x, freqs[i].y);
        // printf("%.3f ", freqs[i]);

    // delete [] freqs;
    // delete [] cuda_samples;
    free(freqs);
    free(cuda_samples);


    return 0;
}
