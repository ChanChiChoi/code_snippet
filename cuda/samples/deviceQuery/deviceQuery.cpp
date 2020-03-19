// cuda_runtime should ahead of helper_cuda
#include<cuda_runtime.h>
#include<helper_cuda.h>

#include<iostream>
#include<iomanip>
//#include<memory>
#include<string>

using std::cout;
using std::endl;
using std::string;

//int *pArgc = nullptr;
//char **pArgv = nullptr;


int main(int argc, char *argv[]){

  cout<<argv[0]<<" starting..."<<endl;
  cout<<"CUDA Device Query(Runtime API) version (CUDART static linking)"<<endl;

  // get number of graphics card
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess){
    cout<<"cudaGetDeviceCount returned:"<<static_cast<int>(error_id)
        <<" -> "<<cudaGetErrorString(error_id)<<endl
        <<"Result = [FAIL]"<<endl;
    exit(EXIT_FAILURE);
  }

  if(deviceCount == 0){
    cout<<"There are no avaliable device(s) that support CUDA"<<endl;
  }else{
    cout<<"Detected "<<deviceCount<<" CUDA Capable device(s)"<<endl;
  }
  // get properties;
  int dev, driverVersion=0, runtimeVersion=0;
  for(dev=0;dev<deviceCount;dev++){
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    cout<<"Device ["<<dev<<"]:                             "
        <<deviceProp.name<<endl;

    //console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout<<" CUDA Driver Version:                           "
        << driverVersion/1000<<"."<<(driverVersion%100)/10<<endl
        <<" Runtime Version:                               "
        << runtimeVersion/1000<<"."<<(runtimeVersion%100)/10<<endl;
    cout<<" CUDA Capability Major/Minor version number:    "
        <<deviceProp.major<<"."<<deviceProp.minor<<endl;

    cout<<" Total amount of global memory:                 "
        <<static_cast<float>(deviceProp.totalGlobalMem)/1048576.0f<<" MBytes "
        <<static_cast<unsigned long long>(deviceProp.totalGlobalMem)<<" bytes"
        <<endl;
    
    cout<<" Multiprocessors:                               "
        <<deviceProp.multiProcessorCount<<endl
        <<" CUDA Cores/MP:                                 "
        <<_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)<<endl
        <<" CUDA Cores:                                    "
        <<_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*
          deviceProp.multiProcessorCount<<endl;
    cout<<" GPU Max Clock rate:                            "
        << deviceProp.clockRate*1e-3f<<" MHz ("
        << deviceProp.clockRate*1e-6f<<" GHz)"
        <<endl;

    cout<<" Memory Clock Rate:                             "
        <<deviceProp.memoryClockRate*1e-3f<<" MHz"<<endl;
    cout<<" Memory Bus Width:                              "
        << deviceProp.memoryBusWidth<<"-bit"<<endl;
    if(deviceProp.l2CacheSize){
      cout<<" L2 Cache Size:                                 "
          <<deviceProp.l2CacheSize<<" bytes"<<endl;
    }
    
    cout<<" Maximum Texture Dimension Size (x,y,z)         "
        <<"1D=("<<deviceProp.maxTexture1D<<"), 2D=("
        <<deviceProp.maxTexture2D[0]<<","
        <<deviceProp.maxTexture2D[1]<<"), 3D=("
        <<deviceProp.maxTexture3D[0]<<","
        <<deviceProp.maxTexture3D[1]<<","
        <<deviceProp.maxTexture3D[2]<<")"<<endl;

    cout<<" Maximum Layered 1D Texture Size, (num) layers  1D=("
        <<deviceProp.maxTexture1DLayered[0]<<"), "
        <<deviceProp.maxTexture1DLayered[1]<<" layers"<<endl;
    cout<<" Maximum Layered 2D Texture Size, (num) layers  2D=("
        <<deviceProp.maxTexture2DLayered[0]<<","
        <<deviceProp.maxTexture2DLayered[1]<<"), "
        <<deviceProp.maxTexture2DLayered[2]<<" layers"<<endl;

    cout<<" Total amount of constant memory:               "
        <<deviceProp.totalConstMem<<" bytes"<<endl;
    cout<<" Total amount of shared memory per block:       "
        <<deviceProp.sharedMemPerBlock<<" bytes"<<endl;
    cout<<" Total number of registers avaliable per block: "
        <<deviceProp.regsPerBlock<<endl;
    cout<<" Warp Size:                                     " 
        <<deviceProp.warpSize<<endl;
    cout<<" Maximum number of threads per multiprocessor:  "
        <<deviceProp.maxThreadsPerMultiProcessor<<endl;
    cout<<" Maximum number of threads per block:           "
        <<deviceProp.maxThreadsPerBlock<<endl;
    cout<<" Max dimension size of a thread block (x,y,z):  ("
        <<deviceProp.maxThreadsDim[0]<<","
        <<deviceProp.maxThreadsDim[1]<<","
        <<deviceProp.maxThreadsDim[2]<<")"<<endl;
    cout<<" Max dimension size of a grid size (x,y,z):     ("
        <<deviceProp.maxGridSize[0]<<","
        <<deviceProp.maxGridSize[1]<<","
        <<deviceProp.maxGridSize[2]<<")"<<endl;
    cout<<" Maximum memory pitch:                          "
        <<deviceProp.memPitch<<" bytes"<<endl;
    cout<<" Maximum alignment:                             "
        <<deviceProp.textureAlignment<<" bytes"<<endl;
    cout<<" Concurrent copy and kernel execution:          "
        <<(deviceProp.deviceOverlap?"Yes":"No")<<" with "
        <<deviceProp.asyncEngineCount<<" copy engines(s)"<<endl;
    cout<<" Run time limit on kernels:                     "
        <<(deviceProp.kernelExecTimeoutEnabled?"Yes":"No")<<endl;
    cout<<" Integrated GPU sharing Host Memory:            "
        <<(deviceProp.integrated?"Yes":"No")<<endl;
    cout<<" Support host page-locked memory mapping:       "
        <<(deviceProp.canMapHostMemory?"Yes":"No")<<endl;
    cout<<" Aligment requirement for Surfaces:             "
        <<(deviceProp.surfaceAlignment?"Yes":"No")<<endl;
    cout<<" Device has ECC support:                        "
        <<(deviceProp.ECCEnabled ?"Enabled":"Disabled")<<endl;
    cout<<" Device supports Unified Addressing (UVA):      "
        <<(deviceProp.unifiedAddressing?"Yes":"No")<<endl;
    cout<<" Device supports Compute Preemption:            "
        <<(deviceProp.computePreemptionSupported?"Yes":"No")<<endl;
    cout<<" Supports Cooperative Kernel Launch:            "
        <<(deviceProp.cooperativeLaunch?"Yes":"No")<<endl;
    cout<<" Supports MultiDevice Co-op Kernel Launch:      "
        <<(deviceProp.cooperativeMultiDeviceLaunch?"Yes":"No")<<endl;
    cout<<" Device PCI Domain ID / Bus ID / location ID:   "
        << deviceProp.pciDomainID<<" / "
        << deviceProp.pciBusID<<" / "
        << deviceProp.pciDeviceID<<endl;

    cout<<" Compute Mode:"<<endl;
    string sComputeMode[]= {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown",
        ""};
    cout<<"  <"<<sComputeMode[deviceProp.computeMode]<<">"<<endl;

    // if there are 2 or more GPUs,
    // query to determine whethre RDMA is supported
    if(deviceCount >= 2){
      cudaDeviceProp prop[64];
      int gpuid[64];// find the first two GPUs that can support P2P
      int gpu_p2p_count = 0;

      for(int i=0; i<deviceCount; i++){
        checkCudaErrors(cudaGetDeviceProperties(&prop[i],i));

        // only Fermi,Kepler,Maxwell,Pascall,Volta,Turing and later support P2p
        if(prop[i].major >= 2){
           //this is an array of P2P capable GPUs
           gpuid[gpu_p2p_count++] = i;
        }
      }
  
      // show all the combinations of support P2P GPUs
      int can_access_peer;

      if(gpu_p2p_count >= 2){
        for(int i=0;i<gpu_p2p_count;i++){
          for(int j=0; j<gpu_p2p_count; i++){
            if(gpuid[i] == gpuid[j]) continue;
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i],
                                                    gpuid[j]));
            cout<<"> Peer access from "
                <<prop[gpuid[i]].name<<" (GPU"<<gpuid[i]<<") ->"
                <<prop[gpuid[j]].name<<" (GPU"<<gpuid[j]<<"): "
                <<(can_access_peer?"Yes":"No")<<endl;
          }
        }
      }
    }

  }
  cout<<"Result = PASS"<<endl;
  return 0;
  
}
