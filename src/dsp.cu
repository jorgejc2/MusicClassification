#include "dsp/dsp.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/numpy.h>
#include <iostream>

// namespace py = pybind11;

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

/* checks if memory could not be allocated */
#define mallocErrchk(ans) { mallocAssert((ans), __FILE__, __LINE__); }
inline void mallocAssert(void* pointer, const char *file, int line, bool abort=true) {
    if (pointer == nullptr)
    {
        fprintf(stderr, "mallocAssert: Returns nullptr at %s %d\n", file, line);
        if (abort) exit(1);
    }
}


/*
    Description: This conducts a serial DFT in O(n^2). This should be the slowest function since it does not 
                 have the same runtime as a radix-2 FFt (O(nlogn)) and is also serial. This function is also most 
                 implemented incorrectly.
*/
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

/*
    Description: This is a serial CPU implementation of the FFT with a runtime of O(nlogn)
    Inputs: 
            const float* samples -- series of time samples
            const int num_samples -- number of samples
    Outputs:
            complex<double>* freqs -- computed frequencies from samples
    Returns: 
            None
    Effects:
            None
*/
__host__ void dsp::FFT(const float* samples, complex<double>* freqs, const int num_samples) {

    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index to currently access
    int bit_shift = (int)log2((float)num_samples); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    complex<double>* shmem = (complex<double>*)malloc(num_samples * 2.5 * sizeof(complex<double>)); // acts as shared memory for holding intermediate computations

    /* easier access of 'shared memory' */
    #define in(i0, swi) shmem[swi*num_samples + i0]
    #define twiddle(i0) shmem[2*num_samples + i0]

    /* iterate through what would be every thread */
    for (int tx = 0; tx < num_samples; tx++) {

        /* rearrange smaples into necessary order and form input_idx */
        for(int i = 0; i < 4; i++)
            idx_arr[i] = dsp::reverse_table[(0x000000FF) & (tx >> (i*8))];

        input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
        input_idx = input_idx >> (32 - bit_shift);
        
        /* copy inputs to shared memory */
        if (tx < num_samples)
            in(input_idx, sw) = complex<double>(samples[tx], 0.0); 

        /* only need half the twiddle factors since they are symmetric */
        if (tx < num_samples/2)
            twiddle(tx) = exp((complex<double>(0.0, -2.0) * complex<double>(M_PI, 0.0) * complex<double>(tx, 0.0)) /  complex<double>(num_samples, 0.0));

    }

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    
    for (int i = 0; i < bit_shift; i++) {
        for (int tx = 0; tx < num_samples; tx++) {
                gs_idx = tx % gs;
                /* this is the positive member of the pair*/
                /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
                if ( (float)gs_idx < (1.0*gs/2) ) {
                    pair_tx = tx + (gs/2);
                    twiddle_idx = (int)(((float)gs_idx / gs)*num_samples);
                    in(tx, !sw) = in(tx, sw) + twiddle(twiddle_idx) * in(pair_tx, sw);
                }
                /* negative member */
                else {
                    pair_tx = tx - (gs/2);
                    twiddle_idx = (int)((((float)gs_idx - gs/2) / gs)*num_samples);
                    in(tx, !sw) = in(pair_tx, sw) - twiddle(twiddle_idx) * in(tx, sw);
                }
            }
        gs *= 2; // number of elements in a group will double
        sw = !sw; // switch where intermediate values will be placed
        }

    /* store final outputs */
    for (int tx = 0; tx < num_samples; tx++)
        freqs[tx] = in(tx, sw);

    /* finished and return */
    free(shmem);
    
    return;

    #undef in
    #undef twiddle
}

/* 
    Description: Simple vector add to aid in testing if the GPU is set up correctly 
    Inputs:
            float* a -- first vector for summing
            float* b -- second vector for summing
            int n -- dimension of vectors
    Outputs:
            float* out -- vector holding the sums
*/
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

__host__ int dsp::cuFFT(float* samples, cuDoubleComplex* freqs, int num_samples) {

    /* create device pointers */
    float* device_samples;
    cuDoubleComplex* device_freqs;

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_samples*sizeof(cuDoubleComplex)));
    size_t shmemsize = num_samples * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    dim3 blockDim(maxThreads > num_samples ? num_samples : maxThreads, 1, 1);
    dim3 gridDim(ceil((float)num_samples / maxThreads), 1, 1);

    /* kernel invocation */
    FFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, num_samples);

    /* synchronize and copy data back to host */
    cudaDeviceSynchronize();
    cudaMemcpy(freqs, device_freqs, num_samples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    /* free memory */
    cudaFree(device_samples);
    cudaFree(device_freqs);

    return maxThreads;
}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
__global__ void dsp::FFT_Kernel(const float* samples, cuDoubleComplex* __restrict__ freqs, const int num_samples) {
    int tx = threadIdx.x;
    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index each thread is responsible for loading to shared memory 
    int bit_shift = (int)log2f((float)num_samples); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    extern __shared__ cuDoubleComplex shmem []; // will be used to hold the inputs/computations and 'twiddle' factors

    /* defines for simpler access to shared memory */
    #define in(i0, swi) shmem[swi*num_samples + i0]
    #define twiddle(i0) shmem[2*num_samples + i0]

    /* rearrange samples into necessary order in shared memory */
    for(int i = 0; i < 4; i++)
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);
    
    /* copy inputs to shared memory */
    if (tx < num_samples)
        in(input_idx, sw) = make_cuDoubleComplex(samples[tx], 0.0); 

    /* only need half the twiddle factors since they are symmetric */
    if (tx < num_samples/2)
        twiddle(tx) = my_cexp(cuCdiv(cuCmul(cuCmul(make_cuDoubleComplex(0.0, -2.0), make_cuDoubleComplex(M_PI, 0.0)), make_cuDoubleComplex(tx, 0.0)), make_cuDoubleComplex(num_samples, 0.0)));

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    
    if (tx < num_samples) {
        for (int i = 0; i < bit_shift; i++) {
            __syncthreads();
            gs_idx = tx % gs;
            /* this is the positive member of the pair*/
            /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
            if ( (float)gs_idx < (1.0*gs/2) ) {
                pair_tx = tx + (gs/2);
                twiddle_idx = (int)(((float)gs_idx / gs)*num_samples);
                in(tx, !sw) = cuCadd(in(tx, sw), cuCmul(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            /* negative member */
            else {
                pair_tx = tx - (gs/2);
                twiddle_idx = (int)((((float)gs_idx - gs/2) / gs)*num_samples);
                in(tx, !sw) = cuCsub(in(pair_tx, sw), cuCmul(twiddle(twiddle_idx),in(tx, sw)));
            }
            gs *= 2; // number of elements in a group will double
            sw = !sw;
        }
    }

    __syncthreads();

    /* return the magnitude as the final output */
    if (tx < num_samples) 
        freqs[tx] = in(tx, sw);

    #undef in
    #undef twiddle
}

__host__ int dsp::cuSTFT(float* samples, double** freqs, int sample_rate, int num_samples, int NFFT, int noverlap = -1) {

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( num_ffts * step >= num_samples )
        num_ffts--;

    int xns_size = num_ffts * NFFT;
    printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    double* xns = (double*)malloc(xns_size*sizeof(double));
    mallocErrchk(xns);

    /* create device pointers */
    float* device_samples;
    double* device_freqs;

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, xns_size*sizeof(double)));
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    printf("block dim: x.%d, y.%d, z.%d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("grid dim: x.%d, y.%d, z.%d\n", gridDim.x, gridDim.y, gridDim.z);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step);

    /* synchronize and copy data back to host */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaMemcpy(xns, device_freqs, xns_size*sizeof(double), cudaMemcpyDeviceToHost));
    
    /* free memory */
    gpuErrchk(cudaFree(device_samples));
    gpuErrchk(cudaFree(device_freqs));

    *freqs = xns;
   
    return xns_size;
}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
__global__ void dsp::STFT_Kernel(const float* samples, double* __restrict__ freqs, int sample_rate, int step) {
    // NOTE; here num_samples is equivalent to NFFT not that actual number of total samples
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    // int num_ffts = blockDim.x;
    int num_samples = blockDim.x;
    int num_ffts = gridDim.x;
    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index each thread is responsible for loading to shared memory 
    int bit_shift = (int)log2f((float)num_samples); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    extern __shared__ cuDoubleComplex shmem []; // will be used to hold the inputs/computations and 'twiddle' factors

    /* defines for simpler access to shared memory */
    #define in(i0, swi) shmem[swi*num_samples + i0]
    #define twiddle(i0) shmem[2*num_samples + i0]

    /* rearrange samples into necessary order in shared memory */
    for(int i = 0; i < 4; i++)
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);
    
    /* copy inputs to shared memory */
    if (tx < num_samples)
        in(input_idx, sw) = make_cuDoubleComplex(samples[bx*step + tx], 0.0); 

    /* only need half the twiddle factors since they are symmetric */
    if (tx < num_samples/2)
        twiddle(tx) = my_cexp(cuCdiv(cuCmul(cuCmul(make_cuDoubleComplex(0.0, -2.0), make_cuDoubleComplex(M_PI, 0.0)), make_cuDoubleComplex(tx, 0.0)), make_cuDoubleComplex(num_samples, 0.0)));

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    
    if (tx < num_samples) {
        for (int i = 0; i < bit_shift; i++) {
            __syncthreads();
            gs_idx = tx % gs;
            /* this is the positive member of the pair*/
            /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
            if ( (float)gs_idx < (1.0*gs/2) ) {
                pair_tx = tx + (gs/2);
                twiddle_idx = (int)(((float)gs_idx / gs)*num_samples);
                in(tx, !sw) = cuCadd(in(tx, sw), cuCmul(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            /* negative member */
            else {
                pair_tx = tx - (gs/2);
                twiddle_idx = (int)((((float)gs_idx - gs/2) / gs)*num_samples);
                in(tx, !sw) = cuCsub(in(pair_tx, sw), cuCmul(twiddle(twiddle_idx),in(tx, sw)));
            }
            gs *= 2; // number of elements in a group will double
            sw = !sw;
        }
    }

    __syncthreads();

    /* return the magnitude as the final output */
    double abs_in = cuCabs(in(tx, sw));
    if (tx < num_samples) 
        freqs[tx*num_ffts + bx] = 5.0 * log10( (abs_in*abs_in) / (sample_rate*num_samples) );

    #undef in
    #undef twiddle
}

__host__ void dsp::cpy_to_symbol() {
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
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