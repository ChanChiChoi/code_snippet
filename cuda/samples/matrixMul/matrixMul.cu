#include<cassert>
#include<iostream>
#include<cstdio>
#include<memory>
#include<cuda_runtime.h>
#include<helper_cuda.h>

using std::unique_ptr;
using std::cout;
using std::endl;

void constantInit(unique_ptr<float[]>&data, int size, float val){
  for(int i=0; i<size; i++)
    data[i] = val;
}


// C = A * B; C_row = A_row; C_col = B_col;
//|------|
//|    B |
//|  A C |
//|------|
// * wA is A's width and wB is B's width

template<int BLOCK_SIZE> __global__ void
matrixMulKernel(float *C, float *A, float *B, int wA, int wB){
  //《CUDA C Programming Guide》figure 10.
  // 其中C的每个块都是将A的横块与B的竖块相乘累加
  // block and thread index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // A矩阵中该块处理的开始处索引
  // wA*BLOCK_SIZE 表示一行 单块的大小，矩阵A一共被划分成blockDim.y个块
  int aBegin = wA*BLOCK_SIZE*by; 
  int aEnd = aBegin + wA-1;//读取一行,当前线程所需要处理的

  // B矩阵中该块处理的开始处索引
  int bBegin = BLOCK_SIZE*bx;//从左往右第一排子块

  //处理完一个块，需要累加处理另一个BLOCK_SIZE,
  //当前子块中每一个点到下一个子块的位移
  int aBlockStep = BLOCK_SIZE;//横向移动
  int bStep = BLOCK_SIZE*wB;//竖向移动

  float Csub = 0;
  for(int a=aBegin, b=bBegin; a<=aEnd; a+=aBlockStep, b+=bStep){
    //填充A和B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    //当前块每个线程填充
    As[ty][tx] = A[a+wA*ty+tx];//a表示跳过前面的子块行，
    Bs[ty][tx] = B[b+wB*ty+tx];

    __syncthreads();
#pragma unroll
    //当前线程，负责C子块的其中一个点
    for(int k=0; k<BLOCK_SIZE; ++k)
      Csub += As[ty][k]*Bs[k][tx];

    __syncthreads();
  }

  //结果回填,当前块中，每个线程填充C当前子块的一个位置
  int c = wB*BLOCK_SIZE*by + BLOCK_SIZE*bx;//C中当前子块的起始位置
  C[c+wB*ty+tx] = Csub;

}

int matrixMultiply(int argc, char **argv, int block_size, 
                   dim3 &dimsA, dim3 &dimsB){
  //x表示col，y表示row；
  //A:[row,col]=[dimsA.y, dimsA.x];
  //B:[row,col]=[dimsB.y, dimsB.x];
  //allocate host mem
  unsigned int size_A = dimsA.x * dimsA.y;
  unique_ptr<float[]> h_A{new float[size_A]};

  unsigned int size_B = dimsB.x*dimsB.y;
  unique_ptr<float[]> h_B{new float[size_B]};

  //initialize host mem
  float const valB = 0.01f;
  constantInit(h_A,size_A, 1.0f);
  constantInit(h_B,size_B, 0.01f);

  //C:[row col]=[dimsC.y, dimsC.x]=[dimsA.y, dimsB.x]
  dim3 dimsC(dimsB.x, dimsA.y, 1);
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
  //dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
  dim3 grid(dimsC.x / threads.x, dimsC.y / threads.y);

  //------------------
  cudaEvent_t st,ed;
  checkCudaErrors(cudaEventCreate(&st));
  checkCudaErrors(cudaEventCreate(&ed));
  checkCudaErrors(cudaEventRecord(st, 0));

  int nIter = 300;
  for(int j=0; j<nIter; j++){
    if(block_size == 16){
      matrixMulKernel<16><<<grid, threads>>>(d_C.get(), d_A.get(), d_B.get(), dimsA.x, dimsB.x);
    }else{
      matrixMulKernel<32><<<grid, threads>>>(d_C.get(), d_A.get(), d_B.get(), dimsA.x, dimsB.x);
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
  cout<<"Performance= "<<gigaFlops<<" GFlop/s, Time = "<<avg<< " msec, Size="
      <<flopsPerMatrixMul<<" Ops, WorkgroupSize= "<<threads.x*threads.y<<endl;

  checkCudaErrors(cudaMemcpy(h_C.get(), d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

  //check
  //计算gpu得到的结果与cpu结果的绝对值比例
  // |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
//  double eps = 1.e-6;
//  for(int i=0; i<static_cast<int>(dimsC.x*dimsC.y); i++){
//
//    double abs_err = fabs(h_C[i] - h_A[i]);
//  }
  return 0;

}


int
main(int argc, char*argv[]){

  int block_size = 32;
  dim3 dimsA(5*5*block_size, 5*2*block_size, 1);
  dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

  int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);
  exit(matrix_result);

}
