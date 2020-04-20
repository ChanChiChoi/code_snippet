#include<cassert>
#include<cstdio>
#include<iostream>

#include<helper_cuda.h>

using namespace std;

#ifdef __linux

#include<unistd.h>
#include<sched.h>
#include<sys/mman.h>
#include<sys/wait.h>
#include<linux/version.h>

#endif
//-------------value,datatype
#define MAX_DEVICES   8
#define PROCESSES_PRE_DEVICE 1
#define DATA_BUF_SIZE 4096

typedef struct ipcDevices_st{
  int count;
  int ordinals[MAX_DEVICES];
}ipcDevices_t;

typedef struct ipcBarrier_st{
  int count;
  bool sense;
  bool allExit;
}ipcBarrier_t;

typedef struct ipcCUDA_st{
  int device;
  pid_t pid;
  cudaIpcEventHandle_t eventHandle;
  cudaIpcMemHandle_t memHandle;
}ipcCUDA_t;

ipcBarrier_t* g_barrier = nullptr;
bool g_procSense;
int g_processCount;

//-----------function
inline bool IsAppBuiltAs64(){
  return sizeof(void*) == 8;
}

void getDeviceCount(ipcDevices_t * devices){
  //因为在fork之前初始化CUDA,会导致驱动上下文出问题，所以先fork

  pid_t pid = fork();
  if(0 == pid){
    int i;
    int count, uvaCount = 0;
    int uvaOrdinals[MAX_DEVICES];
    cout<<"检测多个GPUs..."<<endl;
    checkCudaErrors(cudaGetDeviceCount(&count));
    cout<<"CUDA-capable device 个数:"<<count<<endl;

    cout<<"搜索 UVA capable devices..."<<endl;
  }
}


int
main(int argc, char *argv[]){
  
//---------------------预检查
#if CUDART_VERSION >= 4010 && defined(__linux)
  if(!IsAppBuiltAs64()){
    cout<<argv[0]<<" only supported on 64-bit Linux Os and the app \
must built as a 64-bit target."<<endl;
    exit(EXIT_FAILURE);
  }  
  cout<<"CUDA Version is OK"<<endl;
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
  cout<<argv[0]<<" only support thwi Linux OS Kernel version 2.6.18 and higher"<<endl;
  exit(EXIT_FAILURE);
#endif
//----------------------ipc准备
  ipcDevices_t* s_devices = static_cast<ipcDevices_t*>(mmap(NULL,sizeof(*s_devices),
                                                           PROT_READ | PROT_WRITE,
                                                           MAP_SHARED | MAP_ANONYMOUS, 0, 0 )); 
  assert(MAP_FAILED != s_devices);

  //不能在fork之前初始化CUDA，所以必须先spawn一个进程先
  getDeviceCount(s_devices);  
  
  if(s_devices->count < 1){
    cout<<"需要GPU，且SM大于2的"<<endl;
    exit(EXIT_SUCCESS);
  }else if(s_devices->count > 1)
    g_processCount = PROCESSES_PRE_DEVICE * s_devices->count;
  else
    g_processCount = 2;//一个设备2个进程

  g_barrier = static_cast<ipcBarrier_t*>(mmap(NULL,sizeof(*g_barrier),
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED | MAP_ANONYMOUS, 0, 0 )); 
  assert(MAP_FAILED != g_barrier); 

  memset((void*)g_barrier, 0, sizeof(*g_barrier));
  // 设置local barrier sense flag
  g_procSense = 0;

  //声明共享内存 shared meory for CUDA memory an event handlers
  ipcCUDA_t* s_mem = static_cast<ipcCUDA_t*>(mmap(NULL, g_processCount*sizeof(*s_mem),
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_SHARED| MAP_ANONYMOUS, 0, 0));    
  assert(MAP_FAILED != s_mem);

  //初始化共享内存 shared memory
}
