#include "dsp/dsp.h"
#include <iostream>

/* checks for CUDA errors */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ int dsp::create_spectogram(vector<float> *ts, int NFFT = 256, int noverlap = -1) {
    if (noverlap < 0)
        noverlap = NFFT / 2;

    int32_t ts_size = (int32_t)ts->size();

    nc::NdArray<int> starts_original = nc::arange<int>(0, ts_size, NFFT - noverlap);
    nc::NdArray<int> starts = starts_original[starts_original + NFFT < (int)ts_size];
    /* create a 2D vector where rows represent each time window and columns represent frequency bins */
    vector<vector<float>> xns (starts.size(), vector<float>(ts_size/2));

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

__host__ int dsp::DFT_slow(vector<float> *ts, nc::NdArray<int> *ks, vector<float> *xns, int ts_offset, int NFFT) {
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

/* this will be the serial CPU FFT version which should display speed up over DFT_slow */
__host__ int dsp::FFT(vector<float> *ts, int NFFT, int noverlap) {

    return -1;
}

// #define N 10000000
#define REVERSE_TABLE_SIZE 256

__constant__ unsigned char device_reverse_table[REVERSE_TABLE_SIZE];

__global__ void dsp::vector_add(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    	out[idx] = a[idx] + b[idx];
}

/* calculates the complex float exponent */
__device__ __forceinline__ cuFloatComplex my_cexpf (cuFloatComplex z) {
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

__device__ __forceinline__ cuDoubleComplex my_cexp (cuDoubleComplex z) {
    cuDoubleComplex res;
    double t = exp (z.x);
    sincos (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

__host__ void dsp::FFT_Setup(float* samples, cuDoubleComplex* freqs, int num_samples) {

    float* device_samples;
    cuDoubleComplex* device_freqs;
    cuDoubleComplex* exps = (cuDoubleComplex*)malloc(num_samples * sizeof(cuDoubleComplex));
    cuDoubleComplex* device_exps;

    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_samples*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMalloc((void**)&device_exps, num_samples*sizeof(cuDoubleComplex)));

    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));

    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    int maxThreads = dsp::get_thread_per_block();

    dim3 blockDim(maxThreads > num_samples ? num_samples : maxThreads, 1, 1);
    dim3 gridDim(ceil((float)num_samples / maxThreads), 1, 1);
    cout<< "blockDim " << blockDim.x << endl;
    cout << "gridDim " << gridDim.x << endl;

    size_t shmemsize = num_samples * 3 * sizeof(cuDoubleComplex);

    printf("Starting kernel with %d samples\n", num_samples);

    FFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, device_exps, num_samples);

    cudaDeviceSynchronize();

    cudaMemcpy(freqs, device_freqs, num_samples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(exps, device_exps, num_samples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++) {
        // cout<< exps[i].x << " " <<exps[i].y << endl;
        printf("%.17f %.17f\n", exps[i].x, exps[i].y);
    }

    free(exps);

    cudaDeviceSynchronize();

    cudaFree(device_samples);
    cudaFree(device_freqs);
    cudaFree(device_exps);

    return;
}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
__global__ void dsp::FFT_Kernel(const float* samples, cuDoubleComplex* __restrict__ freqs, cuDoubleComplex * __restrict__ exps, const int num_samples) {
    int tx = threadIdx.x;
    unsigned char idx_arr[4];
    unsigned int input_idx;
    int bit_shift = (int)log2f((float)num_samples); // also corresponds to number of stages 
    int sw = 0;
    extern __shared__ cuDoubleComplex shmem []; // will be used to hold the inputs and 'twiddle' factors

    #define in(i0, swi) shmem[swi*num_samples + i0]
    #define twiddle(i0) shmem[2*num_samples + i0]

    /* rearrange smaples into necessary order for FFT */
    for(int i = 0; i < 4; i++) {
        // ERROR: cannot access reverse_table directly, must use constant memory
        // idx_arr[i] = dsp::reverse_table[(char)(tx >> (i*8))];
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];
    }

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);
    
    if (tx < num_samples) {
        in(input_idx, sw) = make_cuDoubleComplex(samples[tx], 0.0); 
        twiddle(tx) = my_cexp(cuCdiv(cuCmul(cuCmul(make_cuDoubleComplex(0.0, -2.0), make_cuDoubleComplex(M_PI, 0.0)), make_cuDoubleComplex(tx, 0.0)), make_cuDoubleComplex(num_samples, 0.0)));
    }

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    
    if (tx < num_samples) {
        for (int i = 0; i < bit_shift; i++) {
            __syncthreads();
            gs_idx = tx % gs;
            twiddle_idx = (int)(((float)gs_idx / gs)*num_samples);
            /* this is the positive member of the pair*/
            /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
            if ( gs_idx < (gs/2) ) {
                pair_tx = tx + (gs/2);
                in(tx, !sw) = cuCadd(in(tx, sw), cuCmul(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            /* negative member */
            else {
                pair_tx = tx - (gs/2);
                in(tx, !sw) = cuCsub(in(tx, sw), cuCmul(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            gs *= 2; // number of elements in a group will double
            sw = !sw;
            if (i == 2)
                break;
        }
    }

    __syncthreads();

    /* return the magnitude as the final output */
    if (tx < num_samples) {
        // freqs[tx] = log10(cuCabsf(in(tx)));
        freqs[tx] = in(tx, sw);
        exps[tx] = twiddle(tx);
        // freqs[tx] = cuCabsf(twiddle(tx)); // debug by checking if factors are correct
        // freqs[tx] = 1.0 * input_idx; // debug by checking if input idx is correct
        // for (int i = 0; i < 4; i++) {
        //     freqs[tx * 4 + i] = 1.0 * (unsigned int)idx_arr[i];
        // }
    }

    #undef in
    #undef twiddle
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
        std::cout<<"Size of cuDoubleComplex: "<<sizeof(cuDoubleComplex)<<endl;
    }
}

/* assumes only one device is being utilized */
__host__ int dsp::get_thread_per_block() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return deviceProp.maxThreadsPerBlock;
}

// __host__ int dsp::test_cuda(){
//     float *a, *b, *out;
//     float *d_a, *d_b, *d_out;

//     a = (float*)malloc(sizeof(float) * N);
//     b = (float*)malloc(sizeof(float) * N);
//     out = (float*)malloc(sizeof(float) * N);
//     for (int i = 0; i < N; i++) {
// 	    *(a + i) = 1.0;
// 	    *(b + i) = 1.0;
//     }

//     // Allocate device memory for a
//     cudaMalloc((void**)&d_a, sizeof(float) * N);
//     cudaMalloc((void**)&d_b, sizeof(float) *N);
//     cudaMalloc((void**)&d_out, sizeof(float) *N);

//     // Transfer data from host to device memory
//     cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
//     dim3 gridDim(ceil(1.0*N/1024), 1, 1);
//     dim3 blockDim(1024, 1, 1);
//     vector_add<<<gridDim,blockDim>>>(d_out, d_a, d_b, N);
//     cudaDeviceSynchronize();
//     cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < 10; i++)
// 	    printf("%f ", *(out + i));
   
//     get_device_properties();
//     // Cleanup after kernel execution
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_out);

//     free(a);
//     free(b);
//     free(out);

//     return 1;
// }