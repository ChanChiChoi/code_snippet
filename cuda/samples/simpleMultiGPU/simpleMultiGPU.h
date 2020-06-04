#ifndef SIMPLEMULTIGU_H
#define SIMPLEMULTIGU_H

typedef struct{
  
  int dataN;
  float *h_Data;
  float *h_Sum;
  float *d_Data;
  float *d_Sum;
  float *h_Sum_from_device;

  cudaStream_t stream;

}TGPUplan;

extern "C" 
void reduceKernel(float *d_result, float *d_input,
                  int N, int block, int thread, cudaStream_t &s);

#endif
