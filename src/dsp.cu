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

#define N 10000000
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

__host__ void dsp::FFT_Setup(float* samples, float* freqs, int num_samples) {

    float* device_samples;
    float* device_freqs;

    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));

    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(char)));

    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    int maxThreads = dsp::get_thread_per_block();

    dim3 blockDim(maxThreads > num_samples ? maxThreads : num_samples, 1, 1);
    dim3 gridDim(ceil((float)num_samples / maxThreads), 1, 1);

    size_t shmemsize = num_samples * 2 * sizeof(cuFloatComplex);

    FFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, num_samples);

    cudaDeviceSynchronize();

    cudaMemcpy(freqs, device_freqs, num_samples*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_samples);
    cudaFree(device_freqs);

    return;
}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
__global__ void dsp::FFT_Kernel(float* samples, float* freqs, int num_samples) {
    int tx = threadIdx.x;
    char idx_arr[4];
    int input_idx;
    int bit_shift = log2f(num_samples); // also corresponds to number of stages 
    extern __shared__ cuFloatComplex shmem []; // will be used to hold the inputs and 'twiddle' factors

    #define in(i0) shmem[i0]
    #define twiddle(i0) shmem[num_samples + i0]

    /* rearrange smaples into necessary order for FFT */
    for(int i = 0; i < 4; i++) {
        // ERROR: cannot access reverse_table directly, must use constant memory
        // idx_arr[i] = dsp::reverse_table[(char)(tx >> (i*8))];
        idx_arr[i] = device_reverse_table[(char)(tx >> (i*8))];
    }

    input_idx = (int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]) >> bit_shift;

    in(input_idx) = make_cuFloatComplex(samples[tx], 0.0); 
    twiddle(tx) = my_cexpf(cuCdivf(cuCmulf(cuCmulf(cuCmulf(make_cuFloatComplex(0.0, 1.0), make_cuFloatComplex(2.0, 0.0)), make_cuFloatComplex(M_PI, 0.0)), make_cuFloatComplex(tx, 0.0)), make_cuFloatComplex(num_samples, 0.0)));

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    for (int i = 0; i < bit_shift; i++) {
        __syncthreads();
        gs_idx = tx % gs;
        twiddle_idx = (gs_idx / gs)*N;
        /* this is the positive member of the pair*/
        /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
        if ( gs_idx < (gs/2) ) {
            pair_tx = tx + (gs/2);
            in(tx) = cuCaddf(in(tx), cuCmulf(twiddle(twiddle_idx),in(pair_tx)));
        }
        /* negative member */
        else {
            pair_tx = tx - (gs/2);
            in(tx) = cuCsubf(in(tx), cuCmulf(twiddle(twiddle_idx),in(pair_tx)));
        }
        gs *= 2; // number of elements in a group will double
    }

    __syncthreads();

    /* return the magnitude as the final output */
    if (tx < num_samples)
        freqs[tx] = log10(cuCabsf(in(tx)));

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
        std::cout<<"Size of cuFloatComplex: "<<sizeof(cuFloatComplex)<<endl;
    }
}

/* assumes only one device is being utilized */
__host__ int dsp::get_thread_per_block() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return deviceProp.maxThreadsPerBlock;
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