#include<iostream>
#include<stdio.h>
#include<helper_cuda.h>

using namespace std;

#define MAX_DEPTH 16
#define INSERTION_SORT 32



void initialize_data(unsigned int *dst, unsigned int nitems){
  srand(2047);
  for(unsigned i=0; i<nitems; i++)
    dst[i] = rand()%nitems;
}

void run_qsort(unsigned int *data, unsigned int nitems){
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,MAX_DEPTH));

  int left = 0;
  int right = nitems-1;
  cout<<" Launching kernel on the GPU"<<endl;
  cdp_simple_quicksort<<<1,1>>>(data,left,right,0);
  checkCudaErrors(cudaDeviceSynchronize());

}

__global__ void
cdp_simple_quicksort(unsigned int *data, int left, int right, int depth){

}


int main(int argc, char *argv[]){

  int num_items = 128;
  bool verbose = false;
  

  //create input data
  unsigned int *h_data = 0;
  unsigned int *d_data = 0;

  cout<<"Initializing data:"<<endl;
  h_data = static_cast<unsigned int *>(malloc(num_items*sizeof(unsigned int)));
  initialize_data(h_data,num_items);

  //allocate gpu mem
  checkCudaErrors(cudaMalloc((void**)&d_data, num_items*sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_data, h_data, num_items*sizeof(unsigned int), cudaMemcpyHostToDevice));

  //execute
  cout<<"Running quicksort on "<<num_items<<" elements"<<endl;
  run_qsort(d_data, num_items);

  //check result
  cout<<"validating results: ";
  check_results(num_items, d_data);

  free(h_data);
  checkCudaErrors(cudaFree(d_data));

  exit(EXIT_SUCCESS);

}
