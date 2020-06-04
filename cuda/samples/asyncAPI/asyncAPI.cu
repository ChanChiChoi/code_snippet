#include<iostream>

using namespace std;

//#include<cuda_runtime.h>

#include<helper_cuda.h>
//#include<helper_functions.h>

__global__ void
increment_kernel(int *g_data, int inc_value){

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[tid] = g_data[tid] + inc_value;

}

bool corrent_output(int *data, int const n, int const x){

  for(int i=0; i<n; i++){
    if(data[i] != x){
       cout<<"Error! data["<<i<<"] = "<<data[i]
           <<", ref = "<<x<<endl;
    }
  }
  return true;
}

int main(int argc, char *argv[]){

  int devID;
  cudaDeviceProp deviceProps;

  cout<<argv[0]<<" - Starting..."<<endl;
  
  char const * tmp1 = *argv;
  char const ** tmp2 =  &tmp1;
  devID = findCudaDevice(argc,tmp2);

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  cout<<"CUDA device: "<<deviceProps.name<<endl;

  //===========================
  int n = 16*1024*1024;
  int nbytes = n*sizeof(int);
  int value = 26;
  
  // host memory
  int *a = 0;
  //checkCudaErrors(cudaMallocHost((void **)&a, nbytes));

  checkCudaErrors(cudaHostAlloc( (void **)&a, nbytes, cudaHostAllocDefault));
  memset(a, 0, nbytes);


  // device memory
  int *d_a = 0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = 512;
  dim3 blocks = n / threads.x;

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  //asnychronously issue work to the gpu(all to stream 0)
  cudaEventRecord(start,0);
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  // have cpu do some work while waiting for stage 1 to finish
  unsigned long int counter=0;
  
  while(cudaEventQuery(stop) == cudaErrorNotReady)
    counter++;

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
  
  // print cpu and gpu times
  cout<<"time spent executing by the gpu: "<<gpu_time<<endl;
  cout<<"cpu execute "<< counter<<" iterations while waiting for gpu to finish"<<endl;

  // check the output for correntness
  bool bFinalRes = corrent_output(a, n, value);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));

  exit(bFinalRes ? EXIT_SUCCESS: EXIT_FAILURE);

}
