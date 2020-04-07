#include<cassert>
#include<cstdio>
#include<memory>
#include<cuda_runtime.h>
#include<helper_cuda.h>

using std::unique_ptr;

void constantInit(unique_ptr<float[]>&data, int size, float val){
  for(int i=0; i<size; i++)
    data[i] = val;
}

int matrixMultiply(int argc, char **argv, int block_size, 
                   dim3 &dimsA, dim3 &dimsB){
  //allocate host mem
  unsigned int size_A = dimsA.x * dimsA.y;
  unique_ptr<float[]> h_A{new float[size_A]};

  unsigned int size_B = dimsB.x*dimsB.y;
  unique_ptr<float[]> h_B{new float[size_B]};

  //initialize host mem
  float const valB = 0.01f;
  constantInit(h_A,size_A, 1.0f);
  constantInit(h_B,size_B, 0.01f);

  dim3 dimsC(dimsB.x, dimsA.y, 1);//?????不应该是dimsA.x, dimsB.y吗,难道是C=B*A
  unsigned int size_C = dimsC.x*dimsC.y;
  unique_ptr<float[]> h_C{new float[size_C]};

  // allocate device mem 
  unsigned int mem_size_A = sizeof(float)*size_A;
  unsigned int mem_size_B = sizeof(float)*size_B;
  unsigned int mem_size_C = sizeof(float)*size_C;
  float *_d_A, *_d_B, *_d_C;
  checkCudaErrors(cudaMalloc((void**)&_d_A, mem_size_A));
  unique_ptr<float,void(*)(float *)> d_A{_d_A, [](float*p){cudaFree(p);}};
  checkCudaErrors(cudaMalloc((void**)&_d_B,mem_size_B));
  unique_ptr<float, void(*)(float*)> d_B{_d_B, [](float*p){cudaFree(p);}};
  checkCudaErrors(cudaMalloc((void**)&_d_C, mem_size_C));
  unique_ptr<float, void(*)(float*)> d_C{_d_C, [](float*p){cudaFree(p);}};

  //cp host to dev
  checkCudaErrors(cudaMemcpy(d_A.get(),h_A.get(),mem_size_A,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B.get(),h_B.get(),mem_size_B,
                             cudaMemcpyHostToDevice));

  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  //------------------
  cudaEvent_T st,ed;
  checkCudaErrors(cudaEventCreate(&st));
  checkCudaErrors(cudaEventCreate(&ed));
  checkCudaErrors(cudaEventRecord(st, 0));

  int nIter = 300;
  for(int j=0; j<nIter; j++){
    if(block_size == 16){
      matrixMulKernel<16><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }else{
      matrixMulKernel<32><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  cudaDeviceSynchronize();
  float total = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&total, st, ed));
  float avg = total / nIter;
  double flopsPerMatrixMul = 2.0*static_cast<double>(dimsA.y)\
                             *static_cast<double>(dimsB.x);
  double gigaFlops = (flopsPerMatrixMul*1.0e-9f) / (avg/1000.f);

}


int
main(int argc, char*argv[]){

  int block_size = 32;
  dim3 dimsA(5*5*block_size, 5*2*block_size, 1);
  dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

  int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);
  exit(matrix_result);

}
