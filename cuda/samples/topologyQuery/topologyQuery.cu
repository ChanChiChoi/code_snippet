#include<iostream>
#include<helper_cuda.h>

using std::cout;
using std::endl;

int
main(int argc, char *argv[]){
  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  for(int device1 = 0; device1<deviceCount; device1++){

    for(int device2 = 0; device2<deviceCount; device2++){
      if(device1 == device2)
        continue;
      int perfRank=0;
      int atomicSupported = 0;
      int accessSupported = 0;

      checkCudaErrors(cudaDeviceGetP2PAttribute(&accessSupported,
                            cudaDevP2PAttrAccessSupported, device1, device2));
      checkCudaErrors(cudaDeviceGetP2PAttribute(&perfRank,
                            cudaDevP2PAttrPerformanceRank, device1, device2));
      checkCudaErrors(cudaDeviceGetP2PAttribute(&atomicSupported,
                            cudaDevP2PAttrNativeAtomicSupported, device1, device2));

      if(accessSupported){
         cout<<"GPU"<<device1<<" <-> GPU"<<device2<<":"<<endl;
         cout<<" * Atomic Supported: "<<(atomicSupported?"Yes":"No")<<endl;
         cout<<" * Perf Rank: "<<perfRank<<endl;
      }
    }
  }
  for(int device = 0; device<deviceCount; device++){
    
      int atomicSupported = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&atomicSupported,
                            cudaDevAttrHostNativeAtomicSupported, device));
      cout<<"GPU"<<device<<" <-> CPU: "<<endl;
      cout<<" * Atomic Supported: "<<(atomicSupported?"Yes":"No")<<endl;
  }

}
