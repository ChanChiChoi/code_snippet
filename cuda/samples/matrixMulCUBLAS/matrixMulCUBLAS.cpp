// SOME PRECAUTIONS:
// 如果是ROW-MAJOR 矩阵相乘 C = A * B,
// 只需要逆序调用CUBLAS的api: cublasSegemm(B, A)!

// CUBLAS 是使用column-major存储的，但是C/C++ 使用row-major存储.
// 当矩阵指针传递给CUBLAS, 内存布局就会从row-major 到 column-major,
// 这相当于一个潜在的转置(transpose).

// 所以对于 row-major的 C/C++矩阵 A, B,就算是最简单的矩阵相乘C = A * B, 
// 也不能直接用输入时候的顺序 cublasSgemm(A, B)，因为存在潜在转置，
// cublasSegemm(A, B) 的结果其实是 A(T) * B(T).
// 如果col(A(T)) != row(B(T)), 也就是 row(A) != col(B),那么 A(T) 和 B(T) 就无法相乘
// 而且如果 A(T) 和 B(T)是可相乘的,那么结果 C 是一个column-based 的cublas矩阵
// 这意味着要想得到C/C++中的 C(T)，就需要额外的转置将结果转成row-based的 C/C++矩阵

// 为了解决这个问题，我们是为了得到 C, 一个row-major的矩阵
// 在cublas格式中，其实是C(T) (因为有个潜在转置).
// C = A * B, 所以 C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// 所以在输入时候，我们不需要额外的转置代码，只需要调整输入位置就行
//
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.


#include<iostream>
#include<memory>
#include<cassert>
#include<iterator>

#include<cuda_runtime.h>
#include<helper_cuda.h> // 要放在cublas_v2.h的前面
#include<cublas_v2.h>

using namespace std;

typedef struct _matrixSize{
  unsigned int uiWA;//A的width
  unsigned int uiHA;//A的height
  unsigned int uiWB;
  unsigned int uiHB;
  unsigned int uiWC;
  unsigned int uiHC;
} sMatrixSize;


void initializeCUDA(int argc, char *argv[], int &devID,
                   int &iSizeMultiple, sMatrixSize & matrix_size){
  int block_size = 32;
  matrix_size.uiWA = 3 * block_size * iSizeMultiple;
  matrix_size.uiHA = 4 * block_size * iSizeMultiple;
  matrix_size.uiWB = 2 * block_size * iSizeMultiple;
  matrix_size.uiHB = 3 * block_size * iSizeMultiple;
  matrix_size.uiWC = 2 * block_size * iSizeMultiple;
  matrix_size.uiHC = 4 * block_size * iSizeMultiple;

  cout<<"MatrixA: "<<matrix_size.uiHA<<","<<matrix_size.uiWA<<endl;
  cout<<"MatrixB: "<<matrix_size.uiHB<<","<<matrix_size.uiWB<<endl;
  cout<<"MatrixC: "<<matrix_size.uiHC<<","<<matrix_size.uiWC<<endl;

  if(matrix_size.uiWA != matrix_size.uiHB ||
     matrix_size.uiHA != matrix_size.uiHC ||
     matrix_size.uiWB != matrix_size.uiWC ){
      cerr<<"ERROR: Matrix sizes  do not match"<<endl;
      exit(EXIT_FAILURE);
  }
}

void randomInit(unique_ptr<float[]>& data, size_t size){
  for(size_t i=0; i<size; i++){
    data[i] = rand()/static_cast<float>(RAND_MAX);
  }
}

void
matrixMulCPU(unique_ptr<float[]>&h_C, unique_ptr<float[]>& h_A,
             unique_ptr<float[]>&h_B,  unsigned int hA, 
             unsigned int wA, unsigned int wB){
  for(size_t i=0; i<hA; i++){
    for(size_t j=0; j<wB; j++){
       double sum=0;
       for(size_t k=0; k<wA; k++){
          double a = h_A[i*wA+k];
          double b = h_B[k*wB+j];
          sum += a*b;
       }
    h_C[i*wB+j] = static_cast<float>(sum);
    }
  }
}


int
matrixMultiply(int argc, char *argv[], int devID, 
               sMatrixSize &matrix_size){

   int block_size = 32;
   srand(2006);

   unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
   unsigned int mem_size_A = sizeof(float) * size_A;

   unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
   unsigned int mem_size_B = sizeof(float) * size_B;

   unique_ptr<float[]>h_A{new float[size_A]};
   unique_ptr<float[]>h_B{new float[size_B]};

   cout<<"随机初始化A B"<<endl;
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
   
   //device
   cout<<"-----------"<<endl;
   unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
   unsigned int mem_size_C = sizeof(float)*size_C;

   unique_ptr<float[]>h_C{new float[size_C]};
   unique_ptr<float[]>h_CUBLAS{new float[size_C]};

   cout<<"申请device 内存"<<endl;
   float *_d_A, *_d_B, *_d_C;
   checkCudaErrors(cudaMalloc((void**)&_d_A, mem_size_A));
   unique_ptr<float,void(*)(float*)> d_A{_d_A, [](float*p){cudaFree(p);}};

   checkCudaErrors(cudaMalloc((void**)&_d_B, mem_size_B));
   unique_ptr<float,void(*)(float*)> d_B{_d_B, [](float*p){cudaFree(p);}};

   checkCudaErrors(cudaMalloc((void**)&_d_C, mem_size_C));
   unique_ptr<float,void(*)(float*)> d_C{_d_C, [](float*p){cudaFree(p);}};

   checkCudaErrors(cudaMemcpy(d_A.get(), h_A.get(), mem_size_A, 
                              cudaMemcpyHostToDevice));   
   checkCudaErrors(cudaMemcpy(d_B.get(), h_B.get(), mem_size_B, 
                              cudaMemcpyHostToDevice));   

   //准备CUBLAS 2.0
   cout<<" computing from CUBLAS"<<endl;
   int nIter = 30;
   {
     float const alpha = 1.0f;
     float const beta = 0.0f;
     cublasHandle_t handle;
     cudaEvent_t st, ed;

     //checkCudaErrors(cublasCreate(&handle));  
     cublasCreate(&handle);  

     //暖机
     cublasSgemm(handle,
                                 CUBLAS_OP_N, //表示不转置
                                 CUBLAS_OP_N, //表示不转置
                                 matrix_size.uiWB,
                                 matrix_size.uiHA,
                                 matrix_size.uiWA,
                                 &alpha,
                                 d_B.get(),
                                 matrix_size.uiWB,
                                 d_A.get(),
                                 matrix_size.uiWA,
                                 &beta,
                                 d_C.get(),
                                 matrix_size.uiWB    
                     );

     checkCudaErrors(cudaEventCreate(&st));
     checkCudaErrors(cudaEventCreate(&ed));
    
     checkCudaErrors(cudaEventRecord(st,0));
     

     for(int i=0; i<nIter; i++){
       //cublas 是column-major

       cublasSgemm(handle,
                                   CUBLAS_OP_N, //表示不转置
                                   CUBLAS_OP_N, //表示不转置
                                   matrix_size.uiWB,
                                   matrix_size.uiHA,
                                   matrix_size.uiWA,
                                   &alpha,
                                   d_B.get(),
                                   matrix_size.uiWB,
                                   d_A.get(),
                                   matrix_size.uiWA,
                                   &beta,
                                   d_C.get(),
                                   matrix_size.uiWB    
                       );

     }
     checkCudaErrors(cudaEventRecord(ed, 0));
     checkCudaErrors(cudaEventSynchronize(ed));

     float msecTotal = 0.0f;
     checkCudaErrors(cudaEventElapsedTime(&msecTotal, st, ed));

     float msecPerMatrixMul = msecTotal / nIter;
     double flopsPerMatrixMul = 2.0*(double)matrix_size.uiHC*\
                                    (double)matrix_size.uiWC*\
                                    (double)matrix_size.uiHB;

     double gigaFlops = (flopsPerMatrixMul*1e-9f) / (msecPerMatrixMul/1000.f);
     cout<<"Performance= "<<gigaFlops<<" GFlops/s, Time= "<<msecPerMatrixMul
         << flopsPerMatrixMul<<" Ops"<<endl;

     checkCudaErrors(cudaMemcpy(h_CUBLAS.get(), d_C.get(), mem_size_C, 
                               cudaMemcpyDeviceToHost));
     cublasDestroy(handle);
   }

   cout<<"computing from host CPU..."<<endl;
   matrixMulCPU(h_C, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA,
                               matrix_size.uiWB); 
   //TODO: h_C, h_CUBLAS做对比

  return 0;
}



int 
main(int argc, char * argv[]){

  cout<<argv[0]<<" Starting..."<<endl;
  int devID = 0, sizeMult = 5;
  sMatrixSize matrix_size;

  cudaSetDevice(1);
  initializeCUDA(argc, argv, devID, sizeMult, matrix_size);  

  int results = matrixMultiply(argc, argv, devID, matrix_size);
//  return results;
}
