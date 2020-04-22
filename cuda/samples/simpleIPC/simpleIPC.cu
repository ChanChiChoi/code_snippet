/*
在MultiKernel中有2个分支，一个是父进程，剩下都是子进程，其中有个barrier，可以如下图形式理解
============================
process0  |  process1     process2
----------|-------------------------
x=1       |  y=2
proBarrier  proBarrier   proBarrier
----------|-------------------------
print(y=2)| print(x=1)
proBarrier  proBarrier   proBarrier
----------|-------------------------
proBarrier  proBarrier   proBarrier
----------|-------------------------
==================================
如上图所示，所谓barrier，就是不同进程基于此，建立栅栏
保证栅栏下面的代码能够访问上面的其他进程的数据。通过共享内存访问
*/
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
#include<sys/syscall.h>

#include<linux/version.h>

#endif
//-------------value,datatype
#define MAX_DEVICES   8
#define PROCESSES_PRE_DEVICE 1
#define DATA_BUF_SIZE 4096

typedef unsigned long lu;

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
bool g_procSense; // 初始化为false
int g_processCount;

//-----------function
pid_t gettid(void){
  return syscall(SYS_gettid);
}
inline bool IsAppBuiltAs64(){
  return sizeof(void*) == 8;
}

void getDeviceCount(ipcDevices_t * devices){
  //因为在fork之前初始化CUDA,会导致驱动上下文出问题，所以先fork

  pid_t pid = fork();
  if(0 == pid){ // 返回0 为子进程
    int i;
    int count, uvaCount = 0;
    int uvaOrdinals[MAX_DEVICES];
    //cout 不是线程安全的
    printf("检测多个GPUs...\n");
    checkCudaErrors(cudaGetDeviceCount(&count));
    printf("CUDA-capable device 个数:%i\n",count);
    printf("搜索 UVA capable devices...\n");

    for(i=0; i<count; i++){
      cudaDeviceProp prop;
      checkCudaErrors(cudaGetDeviceProperties(&prop,i));
      if(prop.unifiedAddressing){
        uvaOrdinals[uvaCount++] = i;
        printf("> GPU %d = %15s is capable of UVA\n",i,prop.name);
      }
      if(prop.computeMode != cudaComputeModeDefault){
        printf("> GPU 设备必须处在 Compute Mode Default\n");
        printf("> 请使用nvidia-smi 去更改Compute Mode为Default\n");
        exit(EXIT_SUCCESS);
      }
    }

    devices->ordinals[0] = uvaOrdinals[0];

    if(uvaCount < 2){
      devices->count = uvaCount; // 0 or 1
      exit(EXIT_SUCCESS);
    }

    //检查是否支持peer 访问
    printf("检查GPU是否支持peer to peer 内存访问\n");

    devices->count=1;
    int canAccessPeer_0i, canAccessPeer_i0;
    for(i = 1; i<uvaCount; i++){
      checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer_0i, 
                                              uvaOrdinals[0],
                                              uvaOrdinals[i]));
      checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer_i0, 
                                              uvaOrdinals[i],
                                              uvaOrdinals[0]));
      if(canAccessPeer_0i*canAccessPeer_i0){
        devices->ordinals[devices->count] = uvaOrdinals[i];
        printf("> Two-way peer access between GPU:[%d] and GPU:[%d], YES\n",
               devices->ordinals[0],
               devices->ordinals[devices->count++]);
      }
      exit(EXIT_SUCCESS);

    }
  }else{
    //父进程负责等待
    int status;
    waitpid(pid, &status, 0);
    assert(!status);
  }
}


void proBarrier(int index){
  // 提供多线程下变量的加减和逻辑运算的原子操作
  // 实测 __sync_add_and_fetch 可以实现多进程之间原子操作；
  int newCount = __sync_add_and_fetch(&g_barrier->count, 1);

  printf("当前:%d; tid=[%lu], pid=%lu %d \n",index, (lu)gettid(),(lu)getpid(),newCount);
  if(newCount == g_processCount){ // 如果是最后一个进程,重置
    g_barrier->count = 0;
    g_barrier->sense = !g_procSense;
  }else{//如果不是最后一个进程，则sense等于g_procSense,进行栅栏
    while(g_barrier->sense == g_procSense)
      if(!g_barrier->allExit){// 即栅栏没释放的时候，让所有线程都空循环，且不占用cpu
       // sched_yield() causes the calling thread to relinquish the CPU.  The
       // thread is moved to the end of the queue for its static priority and a
       // new thread gets to run.
        sched_yield();
      }else
        exit(EXIT_SUCCESS);
  }
  g_procSense = !g_procSense;
}

__global__ void
simpleKernel(int *dst, int *src, int num){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  dst[idx] = src[idx]/num;
}

void MultiKernel(ipcCUDA_t *s_mem, int index){
   //1 - 进程0将引用的buffer装载进GPU0的显存
   //2 - 其他进程使用P2P，在GPU0上发起一个kernel，
   //3 - 进程0检查结果

  int *d_ptr;//device
  int h_refData[DATA_BUF_SIZE];//host
 
  for(int i=0; i<DATA_BUF_SIZE; i++)
    h_refData[i] = rand();
  checkCudaErrors(cudaSetDevice(s_mem[index].device));
  
  if(0 == index){//父进程
    printf("\n父进程，准备运行kernels\n");    
    int h_results[DATA_BUF_SIZE*MAX_DEVICES*PROCESSES_PRE_DEVICE];

    cudaEvent_t event[MAX_DEVICES*PROCESSES_PRE_DEVICE];

    checkCudaErrors(cudaMalloc((void**)&d_ptr, 
                               DATA_BUF_SIZE*g_processCount*sizeof(int) ));
    //device Ipc
    //Gets an interprocess memory handle for an existing device memory allocation.
    // IpcGet意思就是在创建的进程中，将其送到IPC对应的内存上
    checkCudaErrors(cudaIpcGetMemHandle( (cudaIpcMemHandle_t*)&s_mem[0].memHandle,
                                         (void*)d_ptr ));
    checkCudaErrors(cudaMemcpy( (void*)d_ptr,
                                (void*)h_refData,
                                DATA_BUF_SIZE*sizeof(int),
                                cudaMemcpyHostToDevice ));

    // b.2:栅栏，让其他子进程走else分支，完成event handles创建完成
    // cudaEventCreate s_mem[index].eventHandle 其他进程都创建完成
    proBarrier(index);
    
    for(int i=1; i<g_processCount; i++){
      //IpcOpen就是获取其他进程的句柄
      checkCudaErrors(cudaIpcOpenEventHandle(&event[i],
                                             s_mem[i].eventHandle));
    }

    //b.3: 等待所有进程开启事件record和kernel
    proBarrier(index);
    for(int i=1; i<g_processCount; i++)
      checkCudaErrors(cudaEventSynchronize(event[i]));

    //b.5
    proBarrier(index);
    checkCudaErrors(cudaMemcpy(h_results, 
                               d_ptr+DATA_BUF_SIZE,
                               DATA_BUF_SIZE*(g_processCount-1)*sizeof(int), 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_ptr));
    //----------------------------
    printf("检查结果是否正确\n"); 
    for(int p=1; p<g_processCount; p++){
      for(int i=0; i<DATA_BUF_SIZE; i++){
        if(h_refData[i]/(p+1) != h_results[(p-1)*DATA_BUF_SIZE+i]){
           printf("失败:索引:%d, 进程索引:%d, %i, %i\n",
                   i, p, h_refData[i], h_results[(p-1)*DATA_BUF_SIZE+i]);
           g_barrier->allExit = true;
           exit(EXIT_SUCCESS);
        }
      }
    }
    printf("Result: Pass\n");

  }else{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
    // IpcGet 将当前进程中的句柄送到共享内存上
    checkCudaErrors(cudaIpcGetEventHandle( (cudaIpcEventHandle_t*)&s_mem[index].eventHandle,
                                           event ));
    //b.1: 等进程0初始化device显存
    // 对于其他进程运行过程而言，这一步，需要父进程完成device显存初始化
    proBarrier(index);
    // IpcOpen就是获取共享内存上，其他进程创建的句柄 
    checkCudaErrors(cudaIpcOpenMemHandle((void**)&d_ptr,
                                         s_mem[0].memHandle,
                                         cudaIpcMemLazyEnablePeerAccess));
    printf("> Process %3d: 在GPU:%d 上运行kernel，从进程 %d (对应GPU %d)上读写数据\n",
           index, s_mem[index].device, 0, s_mem[0].device);

    dim3 const threads(512,1);
    dim3 const block(DATA_BUF_SIZE/threads.x, 1);
    simpleKernel<<<block, threads>>>(d_ptr+index*DATA_BUF_SIZE,
                                     d_ptr,
                                     index+1);
    checkCudaErrors(cudaEventRecord(event));

    //b.4 
    proBarrier(index);
    // Close memory mapped with cudaIpcOpenMemHandle.
    checkCudaErrors(cudaIpcCloseMemHandle(d_ptr));

    //b.6 等所有子进程完成事件的使用
    proBarrier(index);
    checkCudaErrors(cudaEventDestroy(event));
  }
  cudaDeviceReset();
  
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
#else
  cout<<"需要CUDA 4.1"<<endl;
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

  //因为后面才是多进程的操作，不能在fork之前初始化CUDA，
  // 所以必须先spawn一个进程，在子进程中进行数据获取
  getDeviceCount(s_devices);  
  
  if(s_devices->count < 1){
    cout<<"需要GPU，且SM大于2的"<<endl;
    exit(EXIT_SUCCESS);
  }else if(s_devices->count > 1)
    g_processCount = PROCESSES_PRE_DEVICE * s_devices->count;
  else
    g_processCount = 2;//如果只有一个设备，那就开2个进程

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
  memset((void*)s_mem, 0, g_processCount*sizeof(*s_mem));
  cout<<"Spawning processes and assigning GPUs..."<<endl;

  //index = 0,...,g_processCount-1
  int index=0;
  //分裂 g_processCount-1 个额外的进程
  for(int i=1; i<g_processCount; i++){
    pid_t pid = fork();
    // On success, the PID of the child process is returned in the parent,
    // and 0 is returned in the child.  On failure, -1 is returned in the
    // parent, no child process is created, and errno is set appropriately
    if(0 == pid){//如果子进程创建成功，则index保存0之后的序号
      index = i;
      break;
    }else{
      s_mem[i].pid = pid; // 保存父进程的pid
    }

  }
  // 把UVA可用设备散入进程中（一个进程一个设备）
  // 如果只有一个设备，那就起2个进程
  //这里是处在某个进程中，各自对共享内存的各自部分进行写入
  if(s_devices->count >1)
    s_mem[index].device = s_devices->ordinals[index / PROCESSES_PRE_DEVICE];
  else
    s_mem[index].device = s_mem[1].device = s_devices->ordinals[0];

  cout<<"> 进程(0为父,其他为子进程) "<<index<<" -> GPU "<<s_mem[index].device<<endl;

  MultiKernel(s_mem, index);

  //等待其他子进程结束，就是join
  if(index == 0){
    for(int i=1; i<g_processCount; i++){
       int status;
       waitpid(s_mem[i].pid, &status, 0);
       assert(!status);
    }
    cout<<"Shutting down..."<<endl;
    exit(EXIT_SUCCESS);
  }
}
