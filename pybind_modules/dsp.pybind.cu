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

int test_function () {return 1;}

__host__ vector<complex<double>> pybind_cuFFT(vector<float> samples) {
    int num_samples = samples.size();

    /* create device pointers */
    float* device_samples;
    cuDoubleComplex cu_freqs [num_samples];
    cuDoubleComplex* device_freqs;
    

    printf("First three samples\n");
    for(int i = 0; i < 3; i++)
        printf("%f\n", samples[i]);

    /* initialize empty freqs vector to return */
    vector<complex<double>> freqs(num_samples, complex<double>(0,0));

    /* allocate memory for device and shared memory */
    gpuErrchk(cudaMalloc((void**)&device_samples, num_samples*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&device_freqs, num_samples*sizeof(cuDoubleComplex)));
    size_t shmemsize = num_samples * 2.5 * sizeof(cuDoubleComplex);

    /* copy data to device and constant memory */
    gpuErrchk(cudaMemcpyToSymbol(dsp::device_reverse_table, dsp::reverse_table, REVERSE_TABLE_SIZE*sizeof(unsigned char)));
    gpuErrchk(cudaMemcpy(device_samples, &samples[0], num_samples*sizeof(float), cudaMemcpyHostToDevice));

    /* get max threads per block and create dimensions */
    int maxThreads = dsp::get_thread_per_block();

    dim3 blockDim(maxThreads > num_samples ? num_samples : maxThreads, 1, 1);
    dim3 gridDim(ceil((float)num_samples / maxThreads), 1, 1);

    /* kernel invocation */
    dsp::FFT_Kernel<<<gridDim, blockDim, shmemsize>>>(device_samples, device_freqs, num_samples);

    /* synchronize and copy data back to host */
    cudaDeviceSynchronize();
    cudaMemcpy(&cu_freqs, device_freqs, num_samples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_samples; i++)
        freqs[i] = complex<double>(cu_freqs[i].x, cu_freqs[i].y);
    printf("First three freqs\n");
    for(int i = 0; i < 3; i++)
        printf("%ld %ld\n", real(freqs[i]), imag(freqs[i]));

    /* free memory */
    cudaFree(device_samples);
    cudaFree(device_freqs);

    return freqs;
}

PYBIND11_MODULE(dsp_module, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("get_thread_per_block", &dsp::get_thread_per_block);
  module_handle.def("cuFFT", &pybind_cuFFT, py::return_value_policy::copy);
    module_handle.def("test_func", &test_function);
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