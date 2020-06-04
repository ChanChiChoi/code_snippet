__global__ void
simple_kernel(int const * pIn, int *pOut, int a){
  __shared__ int sData[NUM_THREADS];
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid == 0)
    printf("current inside [func1]\n");

  sData[threadIdx.x] = pIn[tid];
  __syncthreads();
  
  pOut[tid] = sData[threadIdx.x]*a + tid; //different
}

__global__ void
simple_kernel(int2 const * pIn, int *pOut, int a){

  __shared__ int2 sData[NUM_THREADS];
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid == 0)
    printf("current inside [func2]\n");

  sData[threadIdx.x] = pIn[tid];
  __syncthreads();

  pOut[tid] = (sData[threadIdx.x].x + sData[threadIdx.x].y)*a + tid; //different
}

__global__ void
simple_kernel(int const * pIn1, int const * pIn2, int *pOut, int a){

  __shared__ int sData1[NUM_THREADS];
  __shared__ int sData2[NUM_THREADS];
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid == 0)
    printf("current inside [func3]\n");


  sData1[threadIdx.x] = pIn1[tid];
  sData2[threadIdx.x] = pIn2[tid];
  __syncthreads();
  
  pOut[tid] = (sData1[threadIdx.x]+sData2[threadIdx.x])*a + tid;
}
