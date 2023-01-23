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

/*
    Description: Calculates an exponent to the z power where z is a complex float
    Inputs:
        cuFloatComplex z -- the power to raise euler's constant to
    Outputs:
        None
    Returns:
        cuFloatComplex res -- euler's constant raised to z
    Effects:
        None
*/
__device__ __forceinline__ cuFloatComplex my_cexpf (cuFloatComplex z) {
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

/*
    Description: Calculates an exponent to the z power where z is a complex double
    Inputs:
        cuDoubleComplex z -- the power to raise euler's constant to
    Outputs:
        None
    Returns:
        cuDoubleComplex res -- euler's constant raised to z
    Effects:
        None
*/
__device__ __forceinline__ cuDoubleComplex my_cexp (cuDoubleComplex z) {
    cuDoubleComplex res;
    double t = exp (z.x);
    sincos (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

/*
    Description: Host call for FFT calculation
    Inputs:
        float* samples -- time series
        int num_samples -- number of samples
    Outputs:
        cuDoubleComplex* freqs -- resulting frequencies
    Returns:
        int maxThreads -- the maximum sized FFT the GPU can calculate based on resources
    Effects:
        None
*/
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

/*
    Description: FFT GPU Kernel performing actual computation
    Inputs:
        const float* samples -- time series
        const int num_samples -- number of samples
    Outputs:
        cuDoubleComplex* __restrict__ freqs -- final calculated frequencies
    Returns:
        None
    Effects:
        None
*/
__global__ void dsp::FFT_Kernel(const float* samples, cuDoubleComplex* __restrict__ freqs, const int num_samples) {
    int tx = threadIdx.x; // thread ID
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

/*
    Description: Host call for calculating the Short Time Fourier Transform
    Inputs:
        float* samples -- time series
        int sample_rate -- rate at which analog signal was sampled
        int NFFT -- number of samples to be contained in each segment (should be a power of 2 for radix-2 FFT computation)
        int noverlap -- how many samples to be overlapped between segments; if not set then defaults to NFFT / 2
    Outputs:
        double** freqs -- array to be allocated with resulting frequencies
    Effects:
        Allocates memory towards freqs which caller must eventually deallocate
*/
__host__ int dsp::cuSTFT(float* samples, double** freqs, int sample_rate, int num_samples, int NFFT, int noverlap = -1) {

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( num_ffts * step >= num_samples )
        num_ffts--;

    /* allocate array to hold frequencies */
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

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step);

    /* synchronize and copy data back to host */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaMemcpy(xns, device_freqs, xns_size*sizeof(double), cudaMemcpyDeviceToHost));
    
    /* free memory */
    gpuErrchk(cudaFree(device_samples));
    gpuErrchk(cudaFree(device_freqs));

    /* set user pointer */
    *freqs = xns;
   
    /* return size of frequency array */
    return xns_size;
}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
/*
    Description: Short Time Fourier Transform GPU kernel
    Inputs:
        const float* samples -- time series
        int sample_rate -- rate analogous signal was sampled
        int step -- step rate for indexing samples which is NFFT - noverlap
    Outputs:
        double* __restrict__ freqs -- output array for frequencies
    Returns:
        None
    Effects: 
        None
*/
__global__ void dsp::STFT_Kernel(const float* samples, double* __restrict__ freqs, int sample_rate, int step) {

    int tx = threadIdx.x; // thread ID
    int bx = blockIdx.x; // block ID
    int nfft = blockDim.x; // FFT size to compute
    int num_ffts = gridDim.x; // total number of FFT's being computed
    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index each thread is responsible for loading to shared memory 
    int bit_shift = (int)log2f((float)nfft); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    extern __shared__ cuDoubleComplex shmem []; // will be used to hold the inputs/computations and 'twiddle' factors

    /* defines for simpler access to shared memory */
    #define in(i0, swi) shmem[swi*nfft + i0]
    #define twiddle(i0) shmem[2*nfft + i0]

    /* rearrange samples into necessary order in shared memory */
    for(int i = 0; i < 4; i++)
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);
    
    /* copy inputs to shared memory */
    if (tx < nfft)
        in(input_idx, sw) = make_cuDoubleComplex(samples[bx*step + tx], 0.0); 

    /* only need half the twiddle factors since they are symmetric */
    if (tx < nfft/2)
        twiddle(tx) = my_cexp(cuCdiv(cuCmul(cuCmul(make_cuDoubleComplex(0.0, -2.0), make_cuDoubleComplex(M_PI, 0.0)), make_cuDoubleComplex(tx, 0.0)), make_cuDoubleComplex(nfft, 0.0)));

    /* perform FFT in stages */
    int gs = 2; // the size of each DFT being computed, thus N/gs is the number of groups
    int gs_idx; // idx of thread in the group
    int twiddle_idx; // idx of twiddle factor
    int pair_tx; // the thread idx that the current thread must share data with
    
    /* begin computations of stages; each stage has a different number of groups that shared data */
    if (tx < nfft) {
        for (int i = 0; i < bit_shift; i++) {
            __syncthreads();
            gs_idx = tx % gs; // index of thread in its group for its current stage
            /* this is the positive member of the pair*/
            /* NOTE: this will cause divergence, try and see if there is a way to prevent this */
            if ( (float)gs_idx < (1.0*gs/2) ) {
                pair_tx = tx + (gs/2);
                twiddle_idx = (int)(((float)gs_idx / gs)*nfft);
                in(tx, !sw) = cuCadd(in(tx, sw), cuCmul(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            /* negative member of the pair */
            else {
                pair_tx = tx - (gs/2);
                twiddle_idx = (int)((((float)gs_idx - gs/2) / gs)*nfft);
                in(tx, !sw) = cuCsub(in(pair_tx, sw), cuCmul(twiddle(twiddle_idx),in(tx, sw)));
            }
            gs *= 2; // number of elements in a group will double
            sw = !sw; // switch buffer
        }
    }

    __syncthreads();

    /* return Power Spectral Density value of output */
    double abs_in = cuCabs(in(tx, sw)); // absolute value of final output
    if (tx < nfft) 
        freqs[tx*num_ffts + bx] = 5.0 * log10( (abs_in*abs_in) / (sample_rate*nfft) );

    #undef in
    #undef twiddle
}

/*
    Description: Copies reverse_table to constant memory on the GPU; Necessary since external 
    files importing this library are not able to copy the table to constant memory
    Inputs:
        None
    Outputs:
        None
    Returns:
        None
    Effects:
        Utilizes constant memory on device GPU
*/
__host__ void dsp::cpy_to_symbol() {
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
}

/*
    Description: Gets information from GPU devices
    Inputs:
        None
    Outputs:
        None
    Returns:
        None
    Effects:
        None
*/
__host__ void dsp::get_device_properties()
{
    int deviceCount; // number of GPUs connected
    cudaGetDeviceCount(&deviceCount);

    /* Get information for each GPU */
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

/*
    Description: Get the number of maximum threads per block; Assumes only one GPU is connected
    Inputs:
        None
    Outputs:
        None
    Returns:
        int -- max threads per block
    Effects:
        None
*/
__host__ int dsp::get_thread_per_block() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return deviceProp.maxThreadsPerBlock;
}