#include "dsp_pybind/dsp.pybind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

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

int test_function () {return 1;}

__host__ int test_cuda(){
    int N = 10000000;

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
	    *(a + i) = 4.0;
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
    dsp::vector_add<<<gridDim,blockDim>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
	    printf("%f ", *(out + i));
   
    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);

    return 1;
}

__host__ vector<complex<double>> pybind_cuFFT(vector<float> samples) {
    /* NOTE: complex<double> on host seems to cast well with cuDoubleComplex but not sure if always true */
    int num_samples = samples.size();

    /* create device pointers */
    float* device_samples;
    cuDoubleComplex* device_freqs;

    /* initialize empty freqs vector to return */
    vector<complex<double>> freqs(num_samples, complex<double>(0,0));

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_samples*sizeof(cuDoubleComplex)));
    size_t shmemsize = num_samples * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    dsp::cpy_to_symbol();
    gpuErrchk(cudaMemcpy(device_samples, &samples[0], num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    dim3 blockDim(maxThreads > num_samples ? num_samples : maxThreads, 1, 1);
    dim3 gridDim(ceil((float)num_samples / maxThreads), 1, 1);

    /* kernel invocation */
    dsp::FFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, num_samples);

    /* synchronize and copy data back to host */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(cudaMemcpy(&freqs[0], device_freqs, num_samples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    /* free memory */
    gpuErrchk(cudaFree(device_samples));
    gpuErrchk(cudaFree(device_freqs));

    return freqs;
}

__host__ vector<vector<complex<double>>> pybind_cuSTFT(vector<float> samples, int NFFT, int noverlap) {

    /* initialization */
    int num_samples = samples.size(); // get number of samples

    /* default noverlap */
    if (noverlap < 0)
        noverlap = NFFT / 2;

    int step = NFFT - noverlap;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( num_ffts * step >= num_samples )
        num_ffts--;

    int xns_size = num_ffts * NFFT;
    vector<vector<complex<double>>> xns(NFFT, vector<complex<double>>(num_ffts, complex<double>(0,0)));
    cuDoubleComplex* freqs = (cuDoubleComplex*)malloc(xns_size*sizeof(cuDoubleComplex));
    mallocErrchk(freqs);
    /* create device pointers */
    float* device_samples;
    cuDoubleComplex* device_freqs;

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, xns_size*sizeof(cuDoubleComplex)));
    size_t shmemsize = NFFT * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    dsp::cpy_to_symbol();
    gpuErrchk(cudaMemcpy(device_samples, &samples[0], num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    // Set dimensions
    dim3 blockDim(maxThreads > NFFT ? NFFT : maxThreads, 1, 1);
    dim3 gridDim(num_ffts, 1, 1);

    // printf("block dim: x.%d, y.%d, z.%d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("grid dim: x.%d, y.%d, z.%d\n", gridDim.x, gridDim.y, gridDim.z);

    /* kernel invocation */
    dsp::STFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, NFFT, step);

    /* synchronize and copy data back to host */
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /* may be an issue copying array into a 2D vector */
    gpuErrchk(cudaMemcpy(freqs, device_freqs, xns_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    /* free memory */
    gpuErrchk(cudaFree(device_samples));
    gpuErrchk(cudaFree(device_freqs));

    for (int i = 0; i < NFFT; i++) {
        for (int j = 0; j < num_ffts; j++) {
            xns[i][j] = complex<double>(freqs[i * num_ffts + j].x, freqs[i * num_ffts + j].y);
        }
    }

    free(freqs);

    // for (int i = 0; i < 8; i++) {
    //     printf("%f\n", xns[0][i]);
    // }
   
    return xns;
}

PYBIND11_MODULE(dsp_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("get_thread_per_block", &dsp::get_thread_per_block);
    module_handle.def("cuFFT", &pybind_cuFFT, py::return_value_policy::copy);
    // module_handle.def("cuSTFT", &pybind_cuSTFT, py::return_value_policy::copy);
    module_handle.def("cuSTFT", [](vector<float> samples, int NFFT, int noverlap) {
        printf("len(samples): %d, NFFT: %d, noverlap: %d\n", samples.size(), NFFT, noverlap);
        py::array out = py::cast(pybind_cuSTFT(samples, NFFT, noverlap));
        return out;
    }, py::arg("samples"), py::arg("NFFT"), py::arg("noverlap"), py::return_value_policy::move);
    module_handle.def("test_func", &test_function);
    module_handle.def("test_cuda", &test_cuda);
/* commented out but kept for reference for adding a class */

//   module_handle.def("some_fn_python_name", &some_fn);
//   module_handle.def("some_class_factory", &some_class_factory);
//   py::class_<SomeClass>(
// 			module_handle, "PySomeClass"
// 			).def(py::init<float>())
//     .def_property("multiplier", &SomeClass::get_mult, &SomeClass::set_mult)
//     .def("multiply", &SomeClass::multiply)
//     .def("multiply_list", &SomeClass::multiply_list)
//     // .def_property_readonly("image", &SomeClass::make_image)
//     .def_property_readonly("image", [](SomeClass &self) {
// 				      py::array out = py::cast(self.make_image());
// 				      return out;
// 				    })
//     // .def("multiply_two", &SomeClass::multiply_two)
//     .def("multiply_two", [](SomeClass &self, float one, float two) {
// 			   return py::make_tuple(self.multiply(one), self.multiply(two));
// 			 })
//     .def("function_that_takes_a_while", &SomeClass::function_that_takes_a_while)
//     ;
}