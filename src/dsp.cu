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

        dsp::DFT_slow(ts, &ks, &(xns[m]), starts[m], NFFT);

        // for (int n = 0; n < ts_size/2; n++) {
        //     dcomp a = 0;
        //     float calc = 0.0;
        //     int ts_offset = starts[m];

        //     dcomp curr_n = n;
        //     dcomp curr_NFFT = NFFT;
        //     dcomp curr_pi = M_PI;
        //     dcomp two = 2;

        //     for (int k = 0; k < NFFT; k++) {
        //         dcomp curr_ts = (*ts)[ts_offset + k];
        //         dcomp curr_ks = ks[k];
        //         a += curr_ts * exp((img * two * curr_pi * curr_ks * curr_n)/curr_NFFT);
        //     }

        //     calc = 10*log10(abs(a)*2);

        //     xns[m][n] = calc;
        // }
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

int dsp::DFT_slow(vector<float> *ts, nc::NdArray<int> *ks, vector<float> *xns, int ts_offset, int NFFT) {
    int ts_size = ts->size();
    for (int n = 0; n < ts_size/2; n++) {
            dcomp a = 0;
            float calc = 0.0;

            dcomp curr_n = n;
            dcomp curr_NFFT = NFFT;
            dcomp curr_pi = M_PI;
            dcomp two = 2;

            for (int k = 0; k < NFFT; k++) {
                dcomp curr_ts = (*ts)[ts_offset + k];
                dcomp curr_ks = (*ks)[k];
                a += curr_ts * exp((img * two * curr_pi * curr_ks * curr_n)/curr_NFFT);
            }

            calc = 10*log10(abs(a)*2);

            (*xns)[n] = calc;
    }
    return -1;
}

int dsp::FFT(vector<float> *ts, int NFFT, int noverlap) {

    return -1;
}

#define N 10000000

__global__ void dsp::vector_add(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    	out[idx] = a[idx] + b[idx];
}
__host__ void dsp::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
__host__ int dsp::test_cuda(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
	    *(a + i) = 1.0;
	    *(b + i) = 1.0;
    }

    // Allocate device memory for a
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) *N);
    cudaMalloc((void**)&d_out, sizeof(float) *N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    dim3 gridDim(ceil(1.0*N/1024), 1, 1);
    dim3 blockDim(1024, 1, 1);
    vector_add<<<gridDim,blockDim>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
	    printf("%f ", *(out + i));
   
    get_device_properties();
    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);

    return 1;
}