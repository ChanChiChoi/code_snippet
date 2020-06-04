#include<iostream>
#include<string>
#include<memory>

#include<cuda.h>
#include<helper_cuda_drvapi.h>
#include<drvapi_error_string.h>

using std::cout;
using std::endl;
using std::unique_ptr;
using std::string;

void check(CUresult &error_id, string const &error_str){

  if(error_id != CUDA_SUCCESS){
    cout<<error_str<<" [code]:"<<error_id
        <<" -> [string]:"<<getCudaDrvErrorString(error_id);
    cout<<"Result = FAIL"<<endl;
    exit(EXIT_FAILURE);
  }
}


int
main(int argc, char **argv){

  CUdevice dev;
  int major=0, minor=0;
  int deviceCount=0;
  unique_ptr<char[]> deviceName{new char[1024]};

  cout<<argv[0]<<" Starting..."<<endl;
  // need to link with cuda.lib files on windows os
  cout<<" CUDA Device Query (Driver API) s statically linked version"<<endl;

  CUresult error_id = cuInit(0);
  check(error_id,"cuInit(0) returned ");

  error_id = cuDeviceGetCount(&deviceCount);
  check(error_id,"cuDeviceGetCount returned ");
  
  if(deviceCount==0){
    cout<<"There are no avaliable device(s) that support CUDA"<<endl;
  }else{
    cout<<"Detected "<<deviceCount<<" CUDA Capable device(s)"<<endl;
  }


  for(dev = 0; dev<deviceCount; dev++){
    error_id = cuDeviceComputeCapability(&major, &minor, dev);
    check(error_id, "cuDeviceComputeCapbility returned ");

    error_id = cuDeviceGetName(deviceName.get(), 256, dev);
    check(error_id, "cuDeviceGetName returned ");

    cout<<"Device: "<<dev<<" "<<deviceName.get()<<endl;

    int driverVersion=0;
    cuDriverGetVersion(&driverVersion);
    cout<<"  CUDA Driver Version:                           "
        <<driverVersion/1000<<"."<<(driverVersion%100)/10<<endl;
    cout<<"  CUDA Capability Major/Minor version number:    "
        << major<<"."<<minor<<endl;

    size_t totalGlobalMem;
    error_id = cuDeviceTotalMem(&totalGlobalMem, dev);
    check(error_id,"cuDeviceTotalMem returned ");

    unique_ptr<char> msg{new char[256]};
    cout<<"  Total amount of global memory:                 "
        <<static_cast<float>(totalGlobalMem)/1048576.0f<<" MBytes "
        <<"("<<static_cast<unsigned long long>(totalGlobalMem)<<" bytes)"<<endl;    

    int multiProcessorCount;
    getCudaAttribute<int>(&multiProcessorCount, 
                          CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cout<<"  ("<<multiProcessorCount<<") Multiprocessors, ("
        <<_ConvertSMVer2CoresDRV(major,minor)<<") CUDA Cores/MP:     "
        <<_ConvertSMVer2CoresDRV(major, minor)*multiProcessorCount<<endl;

    int clockRate;
    getCudaAttribute<int>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    cout<<"  GPU Max Clock rate:                            "
        <<clockRate*1e-3f<<" MHz ("<<clockRate*1e-6f<<" GHz)"<<endl;

    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
    cout<<"  Memory Clock rate:                             "
        <<memoryClock*1e-3f<<" MHz"<<endl;

    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    cout<<"  Memory Bus Width:                              "
        <<memBusWidth<<"-bit"<<endl;
    
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
    if(L2CacheSize){
      cout<<"  L2 Cache Size:                                 "
          <<L2CacheSize<<" bytes"<<endl;
    }

    int maxTex1D, maxTex2D[2], maxTex3D[3];
    getCudaAttribute<int>(&maxTex1D, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex2D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex2D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, dev);
    getCudaAttribute<int>(&maxTex3D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex3D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev);
    getCudaAttribute<int>(&maxTex3D[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev);
    cout<<"  Max Texture Dimension Sizes                    "
        <<"1D=("<<maxTex1D<<") 2D=("
        <<maxTex2D[0]<<","<<maxTex2D[1]<<") 3D=("
        <<maxTex3D[0]<<","<<maxTex3D[1]<<","<<maxTex3D[2]<<")"<<endl;

    int maxTex1DLayered[2];
    getCudaAttribute<int>(&maxTex1DLayered[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,dev);
    getCudaAttribute<int>(&maxTex1DLayered[1], 
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, dev);
    cout<<"  Maximum Layered 1D Texture Size, (num) layers  1D=("
        <<maxTex1DLayered[0]<<"), "<<maxTex1DLayered[1]<<" layers"<<endl;

    int maxTex2DLayered[3];
    getCudaAttribute<int>(&maxTex2DLayered[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, dev);
    getCudaAttribute<int>(&maxTex2DLayered[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,dev);
    getCudaAttribute<int>(&maxTex2DLayered[2],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,dev);
    cout<<"  Maximum Layered 2D Texture Size, (num) layers  2D=("
        <<maxTex2DLayered[0]<<","<<maxTex2DLayered[1]<<") "
        <<maxTex2DLayered[2]<<" layers"<<endl;

    int totalConstantMemory;
    getCudaAttribute<int>(&totalConstantMemory, 
                          CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
    cout<<"  Total amount of constant memory:               "
        <<totalConstantMemory<<" bytes"<<endl;
  
    int sharedMemPerBlock;
    getCudaAttribute<int>(&sharedMemPerBlock, 
                          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,dev);
    cout<<"  Total amount of shared memory per block:       "
        <<sharedMemPerBlock<<" bytes"<<endl;

    int regsPerBlock;
    getCudaAttribute<int>(&regsPerBlock, 
                          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
    cout<<"  Total number of registers available per block: "
        <<regsPerBlock<<endl;

    int warpSize;
    getCudaAttribute<int>(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    cout<<"  Warp size:                                     "
        <<warpSize<<endl;
  
    int maxThreadsPerMultiProcessor;
    getCudaAttribute<int>(&maxThreadsPerMultiProcessor,
                          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,dev);
    cout<<"  Maximum number of threads per multiprocessor:  "
        <<maxThreadsPerMultiProcessor<<endl;

    int maxThreadsPerBlock;
    getCudaAttribute<int>(&maxThreadsPerBlock,
                          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    cout<<"  Maximum number of threads per block:           "
        <<maxThreadsPerBlock<<endl;

    int blockDim[3];
    getCudaAttribute<int>(&blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
    getCudaAttribute<int>(&blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
    getCudaAttribute<int>(&blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
    cout<<"  Max dimension size of a thread block (x,y,z):  ("
        <<blockDim[0]<<", "<<blockDim[1]<<", "<<blockDim[2]<<")"<<endl;
  
    int gridDim[3];
    getCudaAttribute<int>(&gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);    
    getCudaAttribute<int>(&gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);    
    getCudaAttribute<int>(&gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);    
    cout<<"  Max dimension size of a grid size (x,y,z):     ("
        <<gridDim[0]<<", "<<gridDim[1]<<", "<<gridDim[2]<<")"<<endl;

    int textureAlign;
    getCudaAttribute<int>(&textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,dev);
    cout<<"  Texture alignment:                             "
        <<textureAlign<<" bytes"<<endl;

    int memPitch;
    getCudaAttribute<int>(&memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, dev);
    cout<<"  Maximum memory pitch:                          "
        <<memPitch<<" bytes"<<endl;
    
    int gpuOverlap;
    getCudaAttribute<int>(&gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);

    int asyncEngineCount;
    getCudaAttribute<int>(&asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,dev);
    cout<<"  Concurrent copy and kernel execution:          "
        <<(gpuOverlap?"Yes":"No")<<" with "<<asyncEngineCount<<" copy engine(s)"<<endl;

    int kernelExecTimeOutEnabled;
    getCudaAttribute<int>(&kernelExecTimeOutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,dev);
    cout<<"  Run time limit on kernels:                     "
        <<(kernelExecTimeOutEnabled?"Yes":"No")<<endl;
    
    int integrated;
    getCudaAttribute<int>(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev );
    cout<<"  Integrated GPU sharing Host Memory:            "
        <<(integrated?"Yes":"No")<<endl;

    int canMapHostMemory;
    getCudaAttribute<int>(&canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
    cout<<"  Support host page-locked memory mapping:       "
        <<(canMapHostMemory?"Yes":"No")<<endl;

    int concurrentKernels;
    getCudaAttribute<int>(&concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
    cout<<"  Concurrent kernel execution:                   "
        <<(concurrentKernels?"Yes":"No")<<endl;

    int surfaceAlignment;
    getCudaAttribute<int>(&surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,dev);
    cout<<"  Alignment requirement for Surfaces:            "
        <<(surfaceAlignment?"Yes":"No")<<endl;
    
    int eccEnabled;
    getCudaAttribute<int>(&eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED,dev);
    cout<<"  Device has ECC support:                        "
        <<(eccEnabled?"Yes":"No")<<endl;

    int unifiedAddressing;
    getCudaAttribute<int>(&unifiedAddressing,CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    cout<<"  Device supports Unified Addressing (UVA):      "
        <<(unifiedAddressing?"Yes":"No")<<endl;

    int cooperativeLaunch;
    getCudaAttribute<int>(&cooperativeLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,dev); 
    cout<<"  Supports MultiDevice Co-op Kernel Launch:      "
        <<(cooperativeLaunch?"Yes":"No")<<endl;

    int pciDomainID, pciBusID, pciDeviceID;
    getCudaAttribute<int>(&pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
    getCudaAttribute<int>(&pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    getCudaAttribute<int>(&pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    cout<<"  Device PCI Domain ID / Bus ID / location ID:   "
        <<pciDomainID<<" / "<<pciBusID<<" / "<<pciDeviceID<<endl;
 

    const string sComputeMode[] =
    {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        ""
    };
    int computeMode;
    getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,dev);
    cout<<"  Compute Mode:"<<endl
        <<"     < "<<sComputeMode[computeMode]<<" >"<<endl;
  }

  // handling more gpu card
  // =========================================
  // if there are more than 1 card, query to determine whether RDMA is supported
  if(deviceCount >=2){
    int gpuid[64];
    int gpu_p2p_count=0;
    int tccDriver=0;

    for(int i=0; i<deviceCount; i++){
      checkCudaErrors(cuDeviceComputeCapability(&major,&minor,i));
      getCudaAttribute<int>(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER,i);

      if(major < 2)
        continue;
      gpuid[gpu_p2p_count++] = i;
    }
    // show all the combinations of support P2P GPUs
    int can_access_peer;
    unique_ptr<char[]> deviceName0{new char[256]};
    unique_ptr<char[]> deviceName1{new char[256]};
    if(gpu_p2p_count >= 2){
      for(int i=0; i<gpu_p2p_count; i++){
        for(int j=0;j<gpu_p2p_count; j++){
         if(gpuid[i]==gpuid[j])
           continue;
         checkCudaErrors(cuDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
         checkCudaErrors(cuDeviceGetName(deviceName0.get(),256, gpuid[i] ));
         checkCudaErrors(cuDeviceGetName(deviceName1.get(),256, gpuid[j]));
         cout<<"> Peer-to-Peer (P2P) access from "<<deviceName0.get()<<" (GPU"<<gpuid[i]<<") -> "
             <<deviceName1.get()<<" (GPU"<<gpuid[j]<<") : "<<(can_access_peer?"Yes":"No")<<endl;
        }
      }

    }






  }
  cout<<"Result = Pass"<<endl;
  exit(EXIT_SUCCESS);

  
}

