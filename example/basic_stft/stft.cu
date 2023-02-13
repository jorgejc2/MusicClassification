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
    /* length of each signal in seconds */
    int length_ts_sec = 3;
    int length_ts1_sec = 1;
    int length_ts2_sec = 3;
    /* length of the total signal and number of total samples */
    int total_ts_length = length_ts_sec + length_ts1_sec + length_ts2_sec;
    int nc_ts_size = sample_rate * 7; // want 7 seconds worth of samples

    /* frequencies to compose the signal from in Hz */
    int freq1 = 697;
    int freq2 = 1209;
    int freq3 = 1336;

    /* Generate samples using NumCPP */
    nc::NdArray<float> lin_freq1 = nc::linspace<float>(0, M_PI*2*freq1, sample_rate);
    nc::NdArray<float> lin_freq2 = nc::linspace<float>(0, M_PI*2*freq2, sample_rate);
    nc::NdArray<float> lin_freq3 = nc::linspace<float>(0, M_PI*2*freq3, sample_rate);

    /* fill a vector with the samples */
    vector<float>nc_ts(nc_ts_size, 0.0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < sample_rate; j++) {
            nc_ts[i*sample_rate + j] = sin(lin_freq1[j]) + sin(lin_freq2[j]);
            nc_ts[i*sample_rate + j + 4000] = sin(lin_freq1[j]) + sin(lin_freq3[j]);
        }
    }

    /* perform simple FFT test on cpu */
    #if (IMPLEMENTATION == 1)
    int fft_size = 1024;

    /* allocate array to hold frequencies and samples */
    complex<double>* freqs = (complex<double>*)malloc(sizeof(complex<double>) * fft_size);
    float* cuda_samples = (float*)malloc(sizeof(float) * fft_size);

    /* fill samples with integers */
    for (int i = 0; i < fft_size; i++)
        cuda_samples[i] = i;

    /* perform and time results */
    auto start = high_resolution_clock::now();
    dsp::FFT(cuda_samples, freqs, fft_size);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    /* print first few results */
    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", real(freqs[i]), imag(freqs[i]));

    /* free memory */
    free(freqs);
    free(cuda_samples);
    #endif

    /* perform simple FFT on gpu */
    #if (IMPLEMENTATION == 2)
    int fft_size = 1024;

    /* allocate frequency array and samples */
    cuDoubleComplex* freqs = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * fft_size);
    float* cuda_samples = (float*)malloc(sizeof(float) * fft_size);

    /* fill samples with integers */
    for (int i = 0; i < fft_size; i++)
        cuda_samples[i] = i;

    /* perform and time results */
    auto start = high_resolution_clock::now();
    dsp::cuFFT(cuda_samples, freqs, fft_size);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    /* print first few results */
    for (int i = 0; i < 8; i++)
        printf("%f + i%f\n", freqs[i].x, freqs[i].y);

    /* free memory */
    free(freqs);
    free(cuda_samples);
    #endif

    /* perform STFT  on cpu ( not yet completed ) */
    #if (IMPLEMENTATION == 3)
    printf("Implementation 3\n");
    #endif

    /* perform STFT on gpu */
    #if (IMPLEMENTATION == 4) 

    int num_samples = nc_ts.size();
    printf("Working with %d samples\n", num_samples);
    int NFFT = 256; // number of samples per segment and their corresponding FFT
    int noverlap = -1; // number of samples to overlap between segments, if -1 then defaults to NFFT / 2

    /* allocate memory for samples and initalize frequency array pointer */
    float* cuda_samples = (float*)malloc(num_samples*sizeof(float));
    mallocErrchk(cuda_samples);
    double* freqs;

    /* fill samples with nc_ts 7 second signal from above */
    for (int i = 0; i < num_samples; i++)
        cuda_samples[i] = nc_ts[i];

    /* perform and time stft */
    printf("Calling cuSTFT\n");
    auto start = high_resolution_clock::now();
    int num_freqs = dsp::cuSTFT(cuda_samples, &freqs, sample_rate, num_samples, NFFT, noverlap, 0, false);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() << endl;

    /* print first few results (NOTE: these are in Power Spectral Density units)*/
    for (int i = 0; i < 8; i++)
        printf("%f\n", freqs[i]);

    /* prints results to a text file */
    // FILE *fp;
    // fp = fopen("../../OutputText/stft_out.txt", "w");

    // int i = 0;
    // int zero_count = 0;
    // printf("%d number of frequencies\n", num_freqs);
    // while (i < num_freqs) {
    //     if (freqs[i] == 0.0)
    //         ++zero_count;

    //     if (i % 256 == 0 && i > 0)
    //         fprintf(fp, "%f \n", freqs[i]);
    //     else
    //         fprintf(fp, "%f ", freqs[i]);

    //     ++i;
    // }    
    // fclose(fp);

    /* free memory */
    free(freqs);
    free(cuda_samples);

    #endif


    return 0;
}
