#include<cstdio>
#include<iostream>
#include<helper_cuda.h>

using std::cout;
using std::endl;

__global__ void
kernel(float *src, float *dst){
  int const idx = blockIdx.x*blockDim.x + threadIdx.x;
  dst[idx] = src[idx]*2.0f;
}

inline bool
isAppBuiltAs64(){
  return sizeof(void*) == 8;
}

int get_GPU(cudaDeviceProp *prop,
            int &nGpu, int gpuId[], int &gpu_count){

  checkCudaErrors(cudaGetDeviceCount(&nGpu));
  cout<<"CUDA-capable device count: "<<nGpu<<endl;
  if(nGpu < 2){
    cout<<"Two or more GPUs with SM 2.0 or higner capability are required"<<endl;
    return -1;
  }

  for(int i=0; i<nGpu; i++){
    checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
    if(prop[i].major >= 2)
      gpuId[gpu_count++] = i;
    cout<<"> GPU:"<<i<<" = "<<prop[i].name<<" capable of Peer-to-Peer (P2P)"<<endl;
  }
  return 0;
}


void check_enable_p2p(cudaDeviceProp prop[],
                      int p2pCapableGPUs[2], int gpu_count, int gpuId[]){

  //------检测p2p
  cout<<"Checking GPU(s) for support of peer to peer memory access..."<<endl;
  int can_access_peer;
  p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

  for(int i=0; i<gpu_count; i++){
    for(int j=0; j<gpu_count; j++){
      if(gpuId[i] == gpuId[j])
        continue;
      checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuId[i], gpuId[j]));
      cout<<"> Peer access from "<<prop[gpuId[i]].name<<"(GPU:"<<gpuId[i]<<") -> "
                                 <<prop[gpuId[j]].name<<"(GPU:"<<gpuId[j]<<")"
                                 <<":"<<(can_access_peer?"Yes":"No")<<endl;
      if(can_access_peer && p2pCapableGPUs[0] == -1){
        p2pCapableGPUs[0] = gpuId[i];
        p2pCapableGPUs[1] = gpuId[j];
      }
    }
  }
  if(p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1){
    cout<<"Two or more GPUs with SM 2.0 or higher capability are required"<<endl;
    cout<<"Peer to Peer access is not available amongst GPUs in the system"<<endl;

    for(int i=0; i<gpu_count; i++)
      checkCudaErrors(cudaSetDevice(gpuId[i]));//是不是有清理的功能
    exit(EXIT_FAILURE);
  }

  //--开启p2p---------
  cout<<"Enabling peer access between GPU:"<<p2pCapableGPUs[0]
      <<" and GPU:"<<p2pCapableGPUs[1]<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[1],0));
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[0],0));


}

void check_UVA(cudaDeviceProp *prop, int gpu0, int gpu1){

  cout<<"Checking GPU:"<<gpu0<<" and GPU:"<<gpu1
      <<" for UVA capabilities..."<<endl;
  bool const has_uva = prop[gpu0].unifiedAddressing &&
                       prop[gpu1].unifiedAddressing;

  cout<<"> "<<prop[gpu0].name<<" (GPU:"<<gpu0
            <<") supports UVA: "<<(prop[gpu0].unifiedAddressing?"Yes":"No")<<endl;
  cout<<"> "<<prop[gpu1].name<<" (GPU:"<<gpu1
            <<") supports UVA: "<<(prop[gpu1].unifiedAddressing?"Yes":"No")<<endl;
  
  if(has_uva)
    cout<<"Both GPUs can support UVA, enabling..."<<endl;
  else
    cout<<"At least one of the two GPUs dose Not support UVA..."<<endl;

}

void check_peer_with_nopeer(int buf_size, int p2pCapableGPUs[2], 
                            float *&d_g0, float *&d_g1, float *&h_c0){

  cout<<"Allocating buffers ("<<int(buf_size/1024.f/1024.f)<<"MB on GPU:"
      <<p2pCapableGPUs[0]<<", GPU:"<<p2pCapableGPUs[1]
      <<" and CPU Host)..."<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
  checkCudaErrors(cudaMalloc(&d_g0, buf_size));
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
  checkCudaErrors(cudaMalloc(&d_g1, buf_size));
  checkCudaErrors(cudaMallocHost(&h_c0, buf_size));

  //创建CUDA event 句柄
  cout<<"Creating event handles..."<<endl;
  cudaEvent_t st, ed;
  float time_memcpy;
  int eventflags = cudaEventBlockingSync;
  checkCudaErrors(cudaEventCreateWithFlags(&st, eventflags));
  checkCudaErrors(cudaEventCreateWithFlags(&ed, eventflags));
 
  checkCudaErrors(cudaEventRecord(st,0));
  //疯狂互相拷贝
  for(int i=0; i<100; i++){
    if(i%2==0)
      checkCudaErrors(cudaMemcpy(d_g1, d_g0, buf_size, cudaMemcpyDefault));
    else
      checkCudaErrors(cudaMemcpy(d_g0, d_g1, buf_size, cudaMemcpyDefault));
  }

  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  checkCudaErrors(cudaEventElapsedTime(&time_memcpy, st, ed));
  // (time_memcpy/1000.f): ms变成s
  // (100.f*buf_size): 来回一共100个buf_size的数据量
  float total_size = (100.f*buf_size) /1024.f/1024.f/1024.f;
  float need_time = time_memcpy/1000.f;
  cout<<"cudaMemcpyPeer / cudaMemcpy between GPU:"<<p2pCapableGPUs[0]
      <<" and GPU:"<<p2pCapableGPUs[1]<<"; "
      <<total_size / need_time <<" GB/s"<<endl;

  checkCudaErrors(cudaEventDestroy(st));
  checkCudaErrors(cudaEventDestroy(ed));
}

void gpu0_from_and_to_gpu1(int buf_size, int p2pCapableGPUs[2],
                           float *&d_g0, float *&d_g1, float *&h_c0){

  cout<<"Preparing host buffer and memory to GPU:"<<p2pCapableGPUs[0]<<endl;
  for(int i=0; i<int(buf_size/sizeof(float));i++ )
    h_c0[i] = float(i%4096);

  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));   
  checkCudaErrors(cudaMemcpy(d_g0, h_c0, buf_size, cudaMemcpyDefault));
  
  dim3 const threads(512);
  dim3 const blocks((buf_size/sizeof(float))/threads.x);
  //在gpu1上执行kernel，读取gpu0的数据，然后将结果写到gpu1
  cout<<"Run kernel on GPU:"<<p2pCapableGPUs[1]<<", taking source data from GPU:"
      <<p2pCapableGPUs[0]<<" and wrigint to GPU:"<<p2pCapableGPUs[1]<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
  kernel<<<blocks, threads>>>(d_g0, d_g1);
  checkCudaErrors(cudaDeviceSynchronize());
  //在gpu0上执行kernel，读取gpu1的数据，然后将结果写到gpu0
  cout<<"Run kernel on GPU:"<<p2pCapableGPUs[0]<<", taking source data from GPU:"
      <<p2pCapableGPUs[1]<<" and wrigint to GPU:"<<p2pCapableGPUs[0]<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
  kernel<<<blocks, threads>>>(d_g1, d_g0);
  checkCudaErrors(cudaDeviceSynchronize());

  //将数据返回到host，然后验证结果
  cout<<"Copy data back to host from GPU:"<<p2pCapableGPUs[0]
      <<" and verify results..."<<endl;
  checkCudaErrors(cudaMemcpy(h_c0, d_g0, buf_size, cudaMemcpyDefault));

  int error_count = 0;
  for(int i=0; i<int(buf_size/sizeof(float)); i++){
    if(h_c0[i] != float(i%4096)*2.0f*2.0f){
      cout<<"Verification error: element:"<<i<<" val="<<h_c0[i]
                                             <<" expection="<<float(i%4096)*2.0f*2.0f<<endl;
      if(error_count++ >10)
        break;
    }
  }

  if(error_count !=0)
    cout<<"Test Failed!"<<endl;
  else
    cout<<"Test Success!"<<endl;
}

void shudown_p2p(int p2pCapableGPUs[2], float *&d_g0, float *&d_g1, float *h_c0, int nGpu){

  cout<<"Disabling peer access..."<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[1]));
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[0]));

  cout<<"shutting down..."<<endl;
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
  checkCudaErrors(cudaFree(d_g0));
  checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
  checkCudaErrors(cudaFree(d_g1));
  checkCudaErrors(cudaFreeHost(h_c0));

  for(int i=0; i<nGpu; i++)
   checkCudaErrors(cudaSetDevice(i));
  
}


int
main(int argc, char *argv[]){

  if(!isAppBuiltAs64()){
    cout<<argv[0]<<"is only supported with on 64-bit OS and the application "
          " must be built as a 64-bit target."<<endl;
  }
  cout<<"checking for multiple GPUS"<<endl;

  //本机一共几张卡
  int nGpu;
  cudaDeviceProp prop[64];//64随便设的，反正一台物理机不会超过这么多卡
  int gpuId[64];
  int gpu_count = 0;
  get_GPU(prop, nGpu, gpuId, gpu_count);

#if CUDART_VERSION >= 4000

  //------检测p2p,开启p2p`
  int p2pCapableGPUs[2];//只找一对能够互相访问的设备
  check_enable_p2p(prop, p2pCapableGPUs, gpu_count, gpuId);

  //----检测2卡的UVA--------
  check_UVA(prop, p2pCapableGPUs[0], p2pCapableGPUs[1]);
  
  //-----速度对比
  size_t const buf_size = 1024*1024*16*sizeof(float);
  float *d_g0 = nullptr;
  float *d_g1 = nullptr;
  float *h_c0 = nullptr;
  //分配buffer到2个gpu，并计算互相拷贝访问时的吞吐量
  check_peer_with_nopeer(buf_size, p2pCapableGPUs, d_g0, d_g1, h_c0);
  //从一个gpu上执行kernel，但是数据来自另一个gpu  
  gpu0_from_and_to_gpu1(buf_size, p2pCapableGPUs, d_g0, d_g1, h_c0);

  //关闭peer 访问
  shudown_p2p(p2pCapableGPUs, d_g0, d_g1, h_c0, nGpu);

#endif

}
