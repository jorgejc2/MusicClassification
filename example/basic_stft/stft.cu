#include "wavreader/wavread.h"
#include "dsp/dsp.h"
#include <math.h>
#include <vector>
#include <NumCpp.hpp>
using namespace std;

#define IMPLEMENTATION 4  // 1: cpu FFT 2: cuda FFT 3: cpu STFT 4: cuda STFT

/* checks if memory could not be allocated */
#define mallocErrchk(ans) { mallocAssert((ans), __FILE__, __LINE__); }
inline void mallocAssert(void* pointer, const char *file, int line, bool abort=true) {
    if (pointer == nullptr)
    {
        fprintf(stderr, "mallocAssert: Returns nullptr, %s %d\n", file, line);
        if (abort) exit(1);
    }
}

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

    // for (int i = 0; i < 10; i++)
    //     printf("%.15f ", nc_ts[i]);
    
    /* if results above match that of Python, then move on to conducting a serial stft */
    // dsp::create_spectogram(&nc_ts, 256, -1);

    /* perform simple FFT test on cpu */
    #if (IMPLEMENTATION == 1)
    int fft_size = 1024;
    complex<double>* freqs = (complex<double>*)malloc(sizeof(complex<double>) * fft_size);
    float* cuda_samples = (float*)malloc(sizeof(float) * fft_size);


    for (int i = 0; i < fft_size; i++)
        cuda_samples[i] = i;

    
    auto start = high_resolution_clock::now();
    dsp::FFT(cuda_samples, freqs, fft_size);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", real(freqs[i]), imag(freqs[i]));

    free(freqs);
    free(cuda_samples);
    #endif

    /* perform simple FFT on gpu */
    #if (IMPLEMENTATION == 2)
    int fft_size = 1024;
    cuDoubleComplex* freqs = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * fft_size);
    float* cuda_samples = (float*)malloc(sizeof(float) * fft_size);

    for (int i = 0; i < fft_size; i++)
        cuda_samples[i] = i;

    auto start = high_resolution_clock::now();
    dsp::cuFFT(cuda_samples, freqs, fft_size);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", freqs[i].x, freqs[i].y);

    free(freqs);
    free(cuda_samples);
    #endif

    /* perform STFT  on cpu */
    #if (IMPLEMENTATION == 3)
    printf("Implementation 3\n");
    #endif

    /* perform STFT on gpu */
    #if (IMPLEMENTATION == 4) 

    // int num_samples = nc_ts.size();
    int num_samples = nc_ts_size;
    printf("Working with %d samples\n", num_samples);
    int NFFT = 256;
    int noverlap = -1;
    float* cuda_samples = (float*)malloc(num_samples*sizeof(float));
    mallocErrchk(cuda_samples);
    cuDoubleComplex* freqs;

    for (int i = 0; i < num_samples; i++)
        cuda_samples[i] = nc_ts[i];

    printf("Calling cuSTFT\n");
    auto start = high_resolution_clock::now();
    int num_freqs = dsp::cuSTFT(cuda_samples, &freqs, num_samples, NFFT, noverlap);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", freqs[i].x, freqs[i].y);

    printf("%d number of frequencies\n", num_freqs);

    free(freqs);
    free(cuda_samples);

    #endif


    return 0;
}
