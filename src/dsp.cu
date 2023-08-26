#include "dsp.h"
#include <float.h> // so that i can get float epsilon
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
// __host__ int dsp::DFT_slow(vector<float> *ts, nc::NdArray<int> *ks, vector<float> *xns, int ts_offset, int NFFT) {
//     int ts_size = ts->size();
//     for (int n = 0; n < ts_size/2; n++) {
//             dcomp a = 0;
//             float calc = 0.0;

//             dcomp curr_n = n;
//             dcomp curr_NFFT = NFFT;
//             dcomp curr_pi = M_PI;
//             dcomp two = 2;

//             for (int k = 0; k < NFFT; k++) {
//                 dcomp curr_ts = (*ts)[ts_offset + k];
//                 dcomp curr_ks = (*ks)[k];
//                 a += curr_ts * exp((img * two * curr_pi * curr_ks * curr_n)/curr_NFFT);
//             }

//             calc = 10*log10(abs(a)*2);

//             (*xns)[n] = calc;
//     }
//     return -1;
// }

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
__host__ int dsp::cuSTFT(float* samples, double** freqs, int sample_rate, int num_samples, int NFFT, int noverlap = -1, bool one_sided = false, int window = 0, bool mag = false) {

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (NFFT - 1) >= num_samples)
        num_ffts--;

    /* allocate array to hold frequencies */
    int one_sided_nfft = NFFT / 2 + 1;
    int xns_size = one_sided ? num_ffts* one_sided_nfft : num_ffts * NFFT;
    // printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    double* xns = (double*)malloc(xns_size*sizeof(double));
    mallocErrchk(xns);

    /* create device pointers */
    float* device_samples;
    double* device_freqs;

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, xns_size*sizeof(double)));
    /* need 2 * NFFT * cuDoubleComplex for alternating buffers that hold computations, 0.5*NFFT*cuDoubleComplex for holding twiddle factors */
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set up stft dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step, window, one_sided, mag);

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

__host__ int dsp::cuSTFT_vector_in(vector<float> &samples, double** freqs, int sample_rate, int NFFT, pair<int,int> &stft_dimensions, int noverlap = -1, bool one_sided = false, int window = 0, bool mag = false) {
    float* samples_ptr = &samples[0];
    int num_samples = samples.size();
    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (NFFT - 1) >= num_samples)
        num_ffts--;

    /* allocate array to hold frequencies */
    int one_sided_nfft = NFFT / 2 + 1;
    int xns_size = one_sided ? num_ffts* one_sided_nfft : num_ffts * NFFT;
    stft_dimensions.first = one_sided ? one_sided_nfft : NFFT;
    stft_dimensions.second = num_ffts;
    // printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    double* xns = (double*)malloc(xns_size*sizeof(double));
    mallocErrchk(xns);

    /* create device pointers */
    float* device_samples;
    double* device_freqs;

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, xns_size*sizeof(double)));
    /* need 2 * NFFT * cuDoubleComplex for alternating buffers that hold computations, 0.5*NFFT*cuDoubleComplex for holding twiddle factors */
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, samples_ptr, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set up stft dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step, window, one_sided, mag);

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
        itn window -- the window to be applied to the input samples;
                      0 = boxcar, 1 = hamming, 2 = hanning
    Outputs:
        double* __restrict__ freqs -- output array for frequencies
    Returns:
        None
    Effects: 
        None
*/
__global__ void dsp::STFT_Kernel(const float* samples, double* __restrict__ freqs, int sample_rate, int step, int window, bool one_sided, bool mag) {

    int tx = threadIdx.x; // thread ID
    int bx = blockIdx.x; // block ID
    int nfft = blockDim.x; // FFT size to compute
    int num_ffts = gridDim.x; // total number of FFT's being computed
    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index each thread is responsible for loading to shared memory 
    int bit_shift = (int)log2f((float)nfft); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    extern __shared__ cuDoubleComplex shmem []; // will be used to hold the inputs/computations and 'twiddle' factors
    double* window_mem = (double*)shmem;

    /* defines for simpler access to shared memory */
    #define in(i0, swi) shmem[swi*nfft + i0]
    #define twiddle(i0) shmem[2*nfft + i0]
    #define w_in(i0) window_mem[i0]

    /* rearrange samples into necessary order in shared memory */
    for(int i = 0; i < 4; i++)
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);

    /* calculate windows and place into shared memory */
    if (window == 1) {
        w_in(tx) = (0.54 - (0.46 * cospif(2*(1.0*tx/(nfft-1)))));
        __syncthreads();
    }
    else if (window == 2) {
        w_in(tx) = (0.5 * (1 - cospif(2*(1.0*tx/(nfft-1)))));
        __syncthreads();
    }
    
    /* place samples into shared memory and apply windows */
    if (tx < nfft) {
        if (window == 0)
            in(input_idx, sw) = make_cuDoubleComplex(samples[bx*step + tx], 0.0); 
        else 
            in(input_idx, sw) = make_cuDoubleComplex(samples[bx*step + tx] * w_in(tx), 0.0); 
    }
        
        

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
    int one_sided_nfft = nfft / 2 + 1;
    if ((mag == true) && (tx < nfft) && (one_sided == false)) {
        freqs[tx*num_ffts + bx] = abs_in * abs_in;
        return;
    }
    else if ((mag == true) && (tx < one_sided_nfft) && (one_sided == true)) {
        freqs[tx*num_ffts + bx] = abs_in * abs_in;
        return;
    }

    /* calculate window scaling factor */
    double window_scaling_factor = (double)nfft; // defaults to length of fft if no window was applied

    if (window == 1 || window == 2) {
        w_in(tx) = w_in(tx) * w_in(tx);
        for (unsigned int stride = nfft / 2; stride >= 1; stride /= 2) {
            __syncthreads();
            if (tx < stride)
                w_in(tx) += w_in(tx + stride);
        }
        __syncthreads();
        window_scaling_factor = w_in(0); // load final sum as scaling factor
    }
    
    /* load final magnitude into output as a tranposed matrix (rows are frequency bins, columns are windows)*/
    if ((tx < nfft) && (one_sided == false)) 
        // freqs[tx*num_ffts + bx] = 10.0 * log10( (abs_in*abs_in) / (sample_rate*window_scaling_factor) );
        freqs[tx*num_ffts + bx] = 10.0 * log10( sqrt((2*abs_in*abs_in) / (sample_rate*window_scaling_factor)) );
    else if ((tx < one_sided_nfft) && (one_sided == true)) 
        // freqs[tx*num_ffts + bx] = 10.0 * log10( (abs_in*abs_in) / (sample_rate*window_scaling_factor) );
        freqs[tx*num_ffts + bx] = 10.0 * log10( sqrt((2*abs_in*abs_in) / (sample_rate*window_scaling_factor)) );
    

    #undef in
    #undef twiddle
    #undef w_in

}

/* note that the max FFT size is limited to the max number of threads allowed in a thread block */
/*
    Description: Short Time Fourier Transform GPU kernel
    Inputs:
        const float* samples -- time series
        int sample_rate -- rate analogous signal was sampled
        int step -- step rate for indexing samples which is NFFT - noverlap
        itn window -- the window to be applied to the input samples;
                      0 = boxcar, 1 = hamming, 2 = hanning
    Outputs:
        double* __restrict__ freqs -- output array for frequencies
    Returns:
        None
    Effects: 
        None
*/
__global__ void dsp::STFT_Kernel_Float(const float* samples, float* __restrict__ freqs, int sample_rate, int step, int window, bool one_sided, bool mag) {

    int tx = threadIdx.x; // thread ID
    int bx = blockIdx.x; // block ID
    int nfft = blockDim.x; // FFT size to compute
    int num_ffts = gridDim.x; // total number of FFT's being computed
    unsigned char idx_arr[4]; // character array used to create input_idx
    unsigned int input_idx; // sample index each thread is responsible for loading to shared memory 
    int bit_shift = (int)log2f((float)nfft); // also corresponds to number of stages 
    int sw = 0; // flag for alternating computational buffers 
    extern __shared__ cuFloatComplex shmemf []; // will be used to hold the inputs/computations and 'twiddle' factors
    float* window_mem = (float*)shmemf;

    /* defines for simpler access to shared memory */
    #define in(i0, swi) shmemf[swi*nfft + i0]
    #define twiddle(i0) shmemf[2*nfft + i0]
    #define w_in(i0) window_mem[i0]

    /* rearrange samples into necessary order in shared memory */
    for(int i = 0; i < 4; i++)
        idx_arr[i] = device_reverse_table[(0x000000FF) & (tx >> (i*8))];

    input_idx = (unsigned int)(idx_arr[0] << 24 | idx_arr[1] << 16 | idx_arr[2] << 8 | idx_arr[3]);
    input_idx = input_idx >> (32 - bit_shift);

    /* calculate windows and place into shared memory */
    if (window == 1) {
        w_in(tx) = (0.54 - (0.46 * cospif(2*(1.0*tx/(nfft-1)))));
        __syncthreads();
    }
    else if (window == 2) {
        w_in(tx) = (0.5 * (1 - cospif(2*(1.0*tx/(nfft-1)))));
        __syncthreads();
    }
    
    /* place samples into shared memory and apply windows */
    if (tx < nfft) {
        if (window == 0)
            in(input_idx, sw) = make_cuFloatComplex(samples[bx*step + tx], 0.0); 
        else 
            in(input_idx, sw) = make_cuFloatComplex(samples[bx*step + tx] * w_in(tx), 0.0); 
    }
        
        

    /* only need half the twiddle factors since they are symmetric */
    if (tx < nfft/2)
        twiddle(tx) = my_cexpf(cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(0.0, -2.0), make_cuFloatComplex(M_PI, 0.0)), make_cuFloatComplex(tx, 0.0)), make_cuFloatComplex(nfft, 0.0)));


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
                in(tx, !sw) = cuCaddf(in(tx, sw), cuCmulf(twiddle(twiddle_idx),in(pair_tx, sw)));
            }
            /* negative member of the pair */
            else {
                pair_tx = tx - (gs/2);
                twiddle_idx = (int)((((float)gs_idx - gs/2) / gs)*nfft);
                in(tx, !sw) = cuCsubf(in(pair_tx, sw), cuCmulf(twiddle(twiddle_idx),in(tx, sw)));
            }
            gs *= 2; // number of elements in a group will float
            sw = !sw; // switch buffer
        }
    }

    __syncthreads();

    /* return Power Spectral Density value of output */
    float abs_in = cuCabsf(in(tx, sw)); // absolute value of final output
    int one_sided_nfft = nfft / 2 + 1;
    if ((mag == true) && (tx < nfft) && (one_sided == false)) {
        freqs[tx*num_ffts + bx] = abs_in * abs_in;
        return;
    }
    else if ((mag == true) && (tx < one_sided_nfft) && (one_sided == true)) {
        freqs[tx*num_ffts + bx] = abs_in * abs_in;
        return;
    }

    /* calculate window scaling factor */
    float window_scaling_factor = (float)nfft; // defaults to length of fft if no window was applied

    if (window == 1 || window == 2) {
        w_in(tx) = w_in(tx) * w_in(tx);
        for (unsigned int stride = nfft / 2; stride >= 1; stride /= 2) {
            __syncthreads();
            if (tx < stride)
                w_in(tx) += w_in(tx + stride);
        }
        __syncthreads();
        window_scaling_factor = w_in(0); // load final sum as scaling factor
    }
    
    /* load final magnitude into output as a tranposed matrix (rows are frequency bins, columns are windows)*/
    if ((tx < nfft) && (one_sided == false)) 
        // freqs[tx*num_ffts + bx] = 10.0 * log10( (abs_in*abs_in) / (sample_rate*window_scaling_factor) );
        freqs[tx*num_ffts + bx] = 10.0 * log10( sqrt((2*abs_in*abs_in) / (sample_rate*window_scaling_factor)) );
    else if ((tx < one_sided_nfft) && (one_sided == true)) 
        // freqs[tx*num_ffts + bx] = 10.0 * log10( (abs_in*abs_in) / (sample_rate*window_scaling_factor) );
        freqs[tx*num_ffts + bx] = 10.0 * log10( sqrt((2*abs_in*abs_in) / (sample_rate*window_scaling_factor)) );
    

    #undef in
    #undef twiddle
    #undef w_in

}


__host__ int dsp::cuMFCC(float* samples, double** freqs, int sample_rate, int num_samples, int NFFT, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq) {

    /* apply a preemphasis filter on the samples */
    dsp::preemphasis(samples, num_samples, preemphasis_b);

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (NFFT - 1) >= num_samples)
        num_ffts--;

    /* allocate array to hold final output */
    int final_output_size = num_ceps * num_ffts;
    double * host_final_output = (double*)malloc(final_output_size*sizeof(double));
    mallocErrchk(host_final_output);

    /* conduct initializations for Mel Spectrum and DCT */
    float low_freq = 0;
    // float high_freq = 2595*log10(1 + sample_rate/(2.0*700));
    float high_freq = 2595*log10(1 + hz_high_freq/(700.0));

    int num_bins = nfilt + 2;

    float mel_step = (high_freq - low_freq) / (num_bins - 1.0);
    int bins[num_bins];
    float mel_point;
    float hz_point;
    for (int i = 0; i < num_bins; i++) {
        mel_point = i*mel_step;
        hz_point = 700 * ( pow(10.0, mel_point/2595.0) - 1);
        // bins[i] = int( (NFFT+1)*hz_point / sample_rate );
        bins[i] = int( (NFFT+1)*hz_point / (hz_high_freq*2) );
    }

    int fbank_rows = nfilt;
    int fbank_cols = NFFT / 2 + 1;
    /* fbank must be cast to double since stft will return double */
    double fbank[fbank_rows*fbank_cols] = {}; // initialize to zeros
    int f_m_minus;
    int f_m;
    int f_m_plus;

    /* Calculate mel filter banks */
    for (int m = 1; m < nfilt + 1; m++) {
        f_m_minus = bins[m - 1];
        f_m = bins[m];
        f_m_plus = bins[m + 1];

        /* in order to coalesce memory, I store the transpose of fbank*/
        for (int k = f_m_minus; k < f_m; k++) {
            fbank[(m-1)*fbank_cols + k] = (double)(2*(k - bins[m-1])) / (bins[m] - bins[m-1]);
        }
        for (int k = f_m; k < f_m_plus; k++) {
            fbank[(m-1)*fbank_cols + k] = (double)(2*(bins[m+1]-k)) / (bins[m + 1] - bins[m]);
        }
    }

    /* calculate DCT matrix */
    int dct_rows = num_ceps;
    int dct_cols = nfilt;
    double dct[dct_rows*dct_cols];

    for (int n = 0; n < num_ceps; n++) {
        for(int m = 0; m < nfilt; m++) {
            dct[n*dct_cols + m] = cos( (m_pi*n*(m-0.5)) / nfilt);
        }
    } 

    /* create device pointers */
    float* device_samples;
    double* device_freqs;
    double* fbank_device;
    double* dct_device;
    double* device_output_s;
    double* device_output_cep;

    /* allocate memory for device and shared memory */
    // int xns_size = num_ffts * (NFFT / 2 + 1);
    // printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_ffts * (NFFT / 2 + 1)*sizeof(double)));
    /* need 2 * NFFT * cuDoubleComplex for alternating buffers that hold computations, 0.5*NFFT*cuDoubleComplex for holding twiddle factors */
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, samples, num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set up stft dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step, window, true, true);

    /* synchronize and free device samples  */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_samples));

    /* perform matrix multiplcation of signal and filter banks */

    /* allocate and copy filter bank to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_s, nfilt*num_ffts*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&fbank_device, fbank_rows*fbank_cols*sizeof(double)));
    gpuErrchk(cudaMemcpy(fbank_device, &fbank, fbank_rows*fbank_cols*sizeof(double), cudaMemcpyHostToDevice));

    /* multiplies matrices MxK and KxN */
    int M = nfilt; 
    int K = NFFT / 2 + 1; 
    int N = num_ffts; 
    dim3 matrix_dimGrid_1(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_1(TILE_SZ_A, 1, 1);
    matrixRegisterTiling<<<matrix_dimGrid_1, matrix_dimBlock_1>>>(device_output_s, fbank_device, device_freqs, M, K, N, true);

    /* synchronize and free memory */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_freqs));
    gpuErrchk(cudaFree(fbank_device));

    /* perform matrix multiplication of DCT and log10 of signal applied with filter banks */

    /* allocate and copy DCT to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_cep, final_output_size*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dct_device, dct_rows*dct_cols*sizeof(double)));
    gpuErrchk(cudaMemcpy(dct_device, &dct, dct_rows*dct_cols*sizeof(double), cudaMemcpyHostToDevice));
    
    /* set up dimensions */
    M = num_ceps; 
    K = nfilt; 
    N = num_ffts; 
    dim3 matrix_dimGrid_2(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_2(TILE_SZ_A, 1, 1);

    /* second matrix multiplication kernel invocation */
    matrixRegisterTiling<<<matrix_dimGrid_2, matrix_dimBlock_2>>>(device_output_cep, dct_device, device_output_s, M, K, N, false);

    /* synchronize */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /* copy final result to host */
    gpuErrchk(cudaMemcpy(host_final_output, device_output_cep, final_output_size*sizeof(double), cudaMemcpyDeviceToHost));

    /* free memory */
    gpuErrchk(cudaFree(device_output_cep));
    gpuErrchk(cudaFree(dct_device));
    gpuErrchk(cudaFree(device_output_s));
    
    /* set user pointer */
    *freqs = host_final_output;
   
    /* return size of frequency array */
    return final_output_size;
}

__host__ int dsp::cuMFCC_vector_in(vector<float> &samples, double** freqs, int sample_rate, int NFFT, pair<int,int> &mfcc_dimensions, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq) {

    int num_samples = samples.size();
    /* apply a preemphasis filter on the samples */
    dsp::preemphasis(&samples[0], num_samples, preemphasis_b);

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (NFFT - 1) >= num_samples)
        num_ffts--;

    /* allocate array to hold final output */
    int final_output_size = num_ceps * num_ffts;
    mfcc_dimensions.first = num_ceps;
    mfcc_dimensions.second = num_ffts;
    double * host_final_output = (double*)malloc(final_output_size*sizeof(double));
    mallocErrchk(host_final_output);

    /* conduct initializations for Mel Spectrum and DCT */
    float low_freq = 0;
    // float high_freq = 2595*log10(1 + sample_rate/(2.0*700));
    float high_freq = 2595*log10(1 + hz_high_freq/(700.0));
    int num_bins = nfilt + 2;

    float mel_step = (high_freq - low_freq) / (num_bins - 1.0);
    int bins[num_bins];
    float mel_point;
    float hz_point;
    for (int i = 0; i < num_bins; i++) {
        mel_point = i*mel_step;
        hz_point = 700 * ( pow(10.0, mel_point/2595.0) - 1);
        // bins[i] = int( (NFFT+1)*hz_point / sample_rate );
        bins[i] = int( (NFFT+1)*hz_point / (hz_high_freq*2) );
    }

    int fbank_rows = nfilt;
    int fbank_cols = NFFT / 2 + 1;
    /* fbank must be cast to double since stft will return double */
    double fbank[fbank_rows*fbank_cols] = {}; // initialize to zeros
    int f_m_minus;
    int f_m;
    int f_m_plus;

    /* Calculate mel filter banks */
    for (int m = 1; m < nfilt + 1; m++) {
        f_m_minus = bins[m - 1];
        f_m = bins[m];
        f_m_plus = bins[m + 1];

        /* in order to coalesce memory, I store the transpose of fbank*/
        for (int k = f_m_minus; k < f_m; k++) {
            fbank[(m-1)*fbank_cols + k] = (double)(2*(k - bins[m-1])) / (bins[m] - bins[m-1]);
        }
        for (int k = f_m; k < f_m_plus; k++) {
            fbank[(m-1)*fbank_cols + k] = (double)(2*(bins[m+1]-k)) / (bins[m + 1] - bins[m]);
        }
    }

    /* calculate DCT matrix */
    int dct_rows = num_ceps;
    int dct_cols = nfilt;
    double dct[dct_rows*dct_cols];

    for (int n = 0; n < num_ceps; n++) {
        for(int m = 0; m < nfilt; m++) {
            dct[n*dct_cols + m] = cos( (m_pi*n*(m-0.5)) / nfilt);
        }
    } 

    /* create device pointers */
    float* device_samples;
    double* device_freqs;
    double* fbank_device;
    double* dct_device;
    double* device_output_s;
    double* device_output_cep;

    /* allocate memory for device and shared memory */
    // int xns_size = num_ffts * (NFFT / 2 + 1);
    // printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_ffts*(NFFT / 2 + 1)*sizeof(double)));
    /* need 2 * NFFT * cuDoubleComplex for alternating buffers that hold computations, 0.5*NFFT*cuDoubleComplex for holding twiddle factors */
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, &samples[0], num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set up stft dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step, window, true, true);

    /* synchronize and free device samples  */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_samples));

    /* perform matrix multiplcation of signal and filter banks */

    /* allocate and copy filter bank to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_s, nfilt*num_ffts*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&fbank_device, fbank_rows*fbank_cols*sizeof(double)));
    gpuErrchk(cudaMemcpy(fbank_device, &fbank, fbank_rows*fbank_cols*sizeof(double), cudaMemcpyHostToDevice));

    /* multiplies matrices MxK and KxN */
    int M = nfilt; 
    int K = NFFT / 2 + 1; 
    int N = num_ffts; 
    dim3 matrix_dimGrid_1(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_1(TILE_SZ_A, 1, 1);
    matrixRegisterTiling<<<matrix_dimGrid_1, matrix_dimBlock_1>>>(device_output_s, fbank_device, device_freqs, M, K, N, true);

    /* synchronize and free memory */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_freqs));
    gpuErrchk(cudaFree(fbank_device));

    /* perform matrix multiplication of DCT and log10 of signal applied with filter banks */

    /* allocate and copy DCT to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_cep, final_output_size*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dct_device, dct_rows*dct_cols*sizeof(double)));
    gpuErrchk(cudaMemcpy(dct_device, &dct, dct_rows*dct_cols*sizeof(double), cudaMemcpyHostToDevice));
    
    /* set up dimensions */
    M = num_ceps; 
    K = nfilt; 
    N = num_ffts; 
    dim3 matrix_dimGrid_2(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_2(TILE_SZ_A, 1, 1);

    /* second matrix multiplication kernel invocation */
    matrixRegisterTiling<<<matrix_dimGrid_2, matrix_dimBlock_2>>>(device_output_cep, dct_device, device_output_s, M, K, N, false);

    /* synchronize */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /* copy final result to host */
    gpuErrchk(cudaMemcpy(host_final_output, device_output_cep, final_output_size*sizeof(double), cudaMemcpyDeviceToHost));

    /* free memory */
    gpuErrchk(cudaFree(device_output_cep));
    gpuErrchk(cudaFree(dct_device));
    gpuErrchk(cudaFree(device_output_s));
    
    /* set user pointer */
    *freqs = host_final_output;
   
    /* return size of frequency array */
    return final_output_size;
}

__host__ int dsp::cuMFCC_vector_in_float(vector<float> &samples, float** freqs, int sample_rate, int NFFT, pair<int,int> &mfcc_dimensions, int noverlap, int window, float preemphasis_b, int nfilt, int num_ceps, float hz_high_freq) {

    int num_samples = samples.size();
    /* apply a preemphasis filter on the samples */
    dsp::preemphasis(&samples[0], num_samples, preemphasis_b);

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    /* Determine how many FFT's need to be computed */
    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (NFFT - 1) >= num_samples)
        num_ffts--;

    /* allocate array to hold final output */
    int final_output_size = num_ceps * num_ffts;
    mfcc_dimensions.first = num_ceps;
    mfcc_dimensions.second = num_ffts;
    float * host_final_output = (float*)malloc(final_output_size*sizeof(float));
    mallocErrchk(host_final_output);

    /* conduct initializations for Mel Spectrum and DCT */
    float low_freq = 0;
    // float high_freq = 2595*log10(1 + sample_rate/(2.0*700));
    float high_freq = 2595*log10(1 + hz_high_freq/(700.0));
    int num_bins = nfilt + 2;

    float mel_step = (high_freq - low_freq) / (num_bins - 1.0);
    int bins[num_bins];
    float mel_point;
    float hz_point;
    for (int i = 0; i < num_bins; i++) {
        mel_point = i*mel_step;
        hz_point = 700 * ( pow(10.0, mel_point/2595.0) - 1);
        // bins[i] = int( (NFFT+1)*hz_point / sample_rate );
        bins[i] = int( (NFFT+1)*hz_point / (hz_high_freq*2) );
    }

    int fbank_rows = nfilt;
    int fbank_cols = NFFT / 2 + 1;
    /* fbank must be cast to float since stft will return float */
    float fbank[fbank_rows*fbank_cols] = {}; // initialize to zeros
    int f_m_minus;
    int f_m;
    int f_m_plus;

    /* Calculate mel filter banks */
    for (int m = 1; m < nfilt + 1; m++) {
        f_m_minus = bins[m - 1];
        f_m = bins[m];
        f_m_plus = bins[m + 1];

        /* in order to coalesce memory, I store the transpose of fbank*/
        for (int k = f_m_minus; k < f_m; k++) {
            fbank[(m-1)*fbank_cols + k] = (float)(2*(k - bins[m-1])) / (bins[m] - bins[m-1]);
        }
        for (int k = f_m; k < f_m_plus; k++) {
            fbank[(m-1)*fbank_cols + k] = (float)(2*(bins[m+1]-k)) / (bins[m + 1] - bins[m]);
        }
    }

    /* calculate DCT matrix */
    int dct_rows = num_ceps;
    int dct_cols = nfilt;
    float dct[dct_rows*dct_cols];

    for (int n = 0; n < num_ceps; n++) {
        for(int m = 0; m < nfilt; m++) {
            dct[n*dct_cols + m] = cosf( (m_pif*n*(m-0.5)) / nfilt);
        }
    } 

    /* create device pointers */
    float* device_samples;
    float* device_freqs;
    float* fbank_device;
    float* dct_device;
    float* device_output_s;
    float* device_output_cep;

    /* allocate memory for device and shared memory */
    // int xns_size = num_ffts * (NFFT / 2 + 1);
    // printf("xns_size: %d, num_ffts: %d, NFFT: %d\n",xns_size, num_ffts, NFFT);
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_ffts*(NFFT / 2 + 1)*sizeof(float)));
    /* need 2 * NFFT * cufloatComplex for alternating buffers that hold computations, 0.5*NFFT*cufloatComplex for holding twiddle factors */
    size_t shmemsize = NFFT * 2.5 * sizeof(cuFloatComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, &samples[0], num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set up stft dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    /* kernel invocation */
    dsp::STFT_Kernel_Float<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, sample_rate, step, window, true, true);

    /* synchronize and free device samples  */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_samples));

    /* perform matrix multiplcation of signal and filter banks */

    /* allocate and copy filter bank to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_s, nfilt*num_ffts*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&fbank_device, fbank_rows*fbank_cols*sizeof(float)));
    gpuErrchk(cudaMemcpy(fbank_device, &fbank, fbank_rows*fbank_cols*sizeof(float), cudaMemcpyHostToDevice));

    /* multiplies matrices MxK and KxN */
    int M = nfilt; 
    int K = NFFT / 2 + 1; 
    int N = num_ffts; 
    dim3 matrix_dimGrid_1(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_1(TILE_SZ_A, 1, 1);
    matrixRegisterTilingFloat<<<matrix_dimGrid_1, matrix_dimBlock_1>>>(device_output_s, fbank_device, device_freqs, M, K, N, true);

    /* synchronize and free memory */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaFree(device_freqs));
    gpuErrchk(cudaFree(fbank_device));

    /* perform matrix multiplication of DCT and log10 of signal applied with filter banks */

    /* allocate and copy DCT to device memory */
    gpuErrchk(cudaMalloc((void**)&device_output_cep, final_output_size*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dct_device, dct_rows*dct_cols*sizeof(float)));
    gpuErrchk(cudaMemcpy(dct_device, &dct, dct_rows*dct_cols*sizeof(float), cudaMemcpyHostToDevice));
    
    /* set up dimensions */
    M = num_ceps; 
    K = nfilt; 
    N = num_ffts; 
    dim3 matrix_dimGrid_2(ceil(M / (1.0*TILE_SZ_A)), ceil(N / (1.0*TILE_SZ_B)), 1);
    dim3 matrix_dimBlock_2(TILE_SZ_A, 1, 1);

    /* second matrix multiplication kernel invocation */
    matrixRegisterTilingFloat<<<matrix_dimGrid_2, matrix_dimBlock_2>>>(device_output_cep, dct_device, device_output_s, M, K, N, false);

    /* synchronize */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /* copy final result to host */
    gpuErrchk(cudaMemcpy(host_final_output, device_output_cep, final_output_size*sizeof(float), cudaMemcpyDeviceToHost));

    /* free memory */
    gpuErrchk(cudaFree(device_output_cep));
    gpuErrchk(cudaFree(dct_device));
    gpuErrchk(cudaFree(device_output_s));
    
    /* set user pointer */
    *freqs = host_final_output;
   
    /* return size of frequency array */
    return final_output_size;
}

/*
    Description: Applies a preemphasis filter of the z-transform function, H(z) = 1 - b*z^-1
*/
__host__ void dsp::preemphasis(float* samples, int num_samples, float b) {

    float prev_sample = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        samples[i] = samples[i] - b*prev_sample;
        prev_sample = samples[i];
    }

    return;
}

__global__ void dsp::matrixRegisterTiling(double * __restrict__ c, //<! [out] and MxN matrix
                       const double *a,        //<! [in] an MxK matrix
                       const double *b,        //<! [in] an KxN matrix
                       const int M, const int K, const int N, const bool log_calc) {

// Macros for accessing flattened matrices
#define A(i1, i0) a[(i1) * K + (i0)] // this will be the mask
#define B(i1, i0) b[(i1)*N + (i0)]
#define C(i1, i0) c[(i1)*N + (i0)]

  // Shared memory for tiling input B array
  __shared__ double B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

  // Privatization of output variables
  double c_reg[TILE_SZ_B];

  // Initialize output values
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    c_reg[outIdx] = 0;
  }

  const unsigned int i = threadIdx.x / TILE_SZ_B;
  const unsigned int j = threadIdx.x % TILE_SZ_B;

  // Loop over the input tiles
  for (unsigned int tileIdx = 0; tileIdx < ceil(K/(1.0 * TILE_SZ_RATIO)); ++tileIdx) {
    // Load the tile of B into shared memory
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
        B_s[i][j] = B(tileIdx * TILE_SZ_RATIO + i, col + j);
    } else {
        B_s[i][j] = 0;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      double a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
      } else {
        a_reg  = 0;
      }
      // Loop over and update the output elemena_regts assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
    __syncthreads();
  }
  
  double temp;
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
        if (log_calc) {
            /* Check if the value is 0. If it is, take the log10 of epsilong for numerical stability */
            temp = c_reg[outIdx];
            C(row, col + outIdx) = (temp == 0.0) ? log10(FLT_EPSILON) : log10(temp);
        }
        else {
            C(row, col + outIdx) = c_reg[outIdx];
        }
    }
  }

#undef A
#undef B
#undef C
}

__global__ void dsp::matrixRegisterTilingFloat(float * __restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int K, const int N, const bool log_calc) {

// Macros for accessing flattened matrices
#define A(i1, i0) a[(i1) * K + (i0)] // this will be the mask
#define B(i1, i0) b[(i1)*N + (i0)]
#define C(i1, i0) c[(i1)*N + (i0)]

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

  // Privatization of output variables
  float c_reg[TILE_SZ_B];

  // Initialize output values
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    c_reg[outIdx] = 0;
  }

  const unsigned int i = threadIdx.x / TILE_SZ_B;
  const unsigned int j = threadIdx.x % TILE_SZ_B;

  // Loop over the input tiles
  for (unsigned int tileIdx = 0; tileIdx < ceil(K/(1.0 * TILE_SZ_RATIO)); ++tileIdx) {
    // Load the tile of B into shared memory
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
        B_s[i][j] = B(tileIdx * TILE_SZ_RATIO + i, col + j);
    } else {
        B_s[i][j] = 0;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      float a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
      } else {
        a_reg  = 0;
      }
      // Loop over and update the output elemena_regts assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
    __syncthreads();
  }
  
  float temp;
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
        if (log_calc) {
            /* Check if the value is 0. If it is, take the log10 of epsilong for numerical stability */
            temp = c_reg[outIdx];
            C(row, col + outIdx) = (temp == 0.0) ? log10(FLT_EPSILON) : log10(temp);
        }
        else {
            C(row, col + outIdx) = c_reg[outIdx];
        }
    }
  }

#undef A
#undef B
#undef C
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

        std::cout<<"Concurrent Kernels: "<<deviceProp.concurrentKernels<<std::endl;
        std::cout<<"Device Overlap: "<<deviceProp.deviceOverlap<<std::endl;
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