#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<ctime>
#include<vector>
#include<algorithm>

#ifdef USE_PTHREADS
#include<pthread.h>
#else
#include <omp.h>
#endif

#include<cublas_v2.h>
#include<helper_cuda.h>

using std::vector;
using std::cout;
using std::endl;


template<typename T>
struct Task{
  unsigned int size, id;
  T *data;
  T *result;
  T *vector;

  Task():size(0), id(0), data(nullptr), 
         result(nullptr), vector(nullptr){}
  Task(unsigned int s):size(s), id(0), data(nullptr),
                       result(nullptr){
    checkCudaErrors(cudaMallocManaged(&data, sizeof(T)*size*size));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(T)*size));
    checkCudaErrors(cudaMallocManaged(&vector, sizeof(T)*size));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  ~Task(){
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaFree(vector));
  }

  void allocate(unsigned int const s, unsigned int const unique_id){
    id = unique_id;
    size = s;
    checkCudaErrors(cudaMallocManaged(&data, sizeof(T)*size*size));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(T)*size));
    checkCudaErrors(cudaMallocManaged(&vector, sizeof(T)*size));
    checkCudaErrors(cudaDeviceSynchronize());

    for(int i=0; i<size*size; i++)
      data[i] = (T)((float)rand()/RAND_MAX*100);

    for(int i=0; i<size; i++){
      result[i] = 0;
      vector[i] = (T)((float)rand()/RAND_MAX*100);
    }
  }

};

#ifdef USE_PTHREADS
typedef struct threadData_t{
  int tid;
  cudaStream_t *streams;
  cublasHandle_t *handles;
  int taskSize;

  Task<double> *TaskListPtr;
}threadData;
#endif

template<typename T> void 
initialize_tasks(vector<Task<T>> &TaskList){
  for(unsigned int i=0; i<TaskList.size(); i++){
    int size;
    size = max(64, (int)(T)((float)rand()/RAND_MAX));
    TaskList[i].allocate(size,i);
  }
}


template<typename T> void
gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result){
  for(int i=0; i<m; i++){
    result[i] *= beta;
    for(int j=0; j<n; j++)
      result[i] += A[i*n+j]*x[j]; 
  }
}

#ifdef USE_PTHREADS
void execute(void *inpArgs){
  // 一个单独的线程，其中threadData中streams和handles都是整个数组
  threadData *dataPtr = (threadData*)inpArgs;
  cudaStream_t *stream = dataPtr->streams;
  cublasHandle_t *handle = dataPtr->handles;
  int tid = dataPtr->tid;

  for(int i=0; i<dataPtr->taskSize; i++){

    Task<double>&t = dataPtr->TaskListPtr[i];
    // 如果任务很小，就放到0号流中
    // 且用cpu操作，否则才用cublas
    // 一共4个线程，如果分到每个线程都小于100个任务，那么其实整体都在0号流中执行
    cout<<"Task ["<<t.id<<"], thread ["<<tid<<"] executing on host ("<<t.size<<")"<<endl;
    if(t.size < 100){
      // attach managed memory to a (dummy) stream to allow host access while the device is running
      checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
      // 确保异步cudaStreamAttachMemAsync 已经执行了
      checkCudaErrors(cudaStreamSynchronize(stream[0]));
      // call the host operation
      gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    }else{

      double one = 1.0;
      double zero = 0.0;
      checkCudaErrors(cublasSetStream(handle[tid+1], stream[tid+1]));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.data, 0, cudaMemAttachSingle));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.vector, 0, cudaMemAttachSingle));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.result, 0, cudaMemAttachSingle));
      // 调用device的
      checkCudaErrors(cublasDgemv(handle[tid+1], CUBLAS_OP_N, t.size, t.size, &one, t.data, t.size,
                                  t.vector, 1, &zero, t.result,1));

    }
  }
}
#else
template<typename T> void
execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream, int tid){

  cout<<"Task ["<<t.id<<"], thread ["<<tid<<"] executing on host ("<<t.size<<")"<<endl;

  if(t.size<100){
    checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
    // 确保异步cudaStreamAttachMemAsync 已经执行了
    checkCudaErrors(cudaStreamSynchronize(stream[0]));
    // call the host operation
    gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
  }else{
    double one = 1.0;
    double zero = 0.0;
    checkCudaErrors(cublasSetStream(handle[tid+1], stream[tid+1]));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.data, 0, cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.vector, 0, cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid+1], t.result, 0, cudaMemAttachSingle));
    // 调用device的
    checkCudaErrors(cublasDgemv(handle[tid+1], CUBLAS_OP_N, t.size, t.size, &one, t.data, t.size,
                                t.vector, 1, &zero, t.result,1));
  }
}




#endif

int
main(int argc, char *argv[]){
  cudaDeviceProp prop;
  int dev_id = 0;
  checkCudaErrors(cudaGetDeviceProperties(&prop, dev_id));

  if(!prop.managedMemory){
    fprintf(stderr, "Unified Memory not supported on this device\n");
    exit(EXIT_FAILURE);
  }
  if(prop.computeMode == cudaComputeModeProhibited){
    fprintf(stderr, "This sample requires a device in either default or process exclusive mode");
    exit(EXIT_FAILURE);
  }

  int seed = time(NULL);
  srand((unsigned int)seed);
  int const nthreads = 4;
  cudaStream_t *streams = new cudaStream_t[nthreads+1]; // streams[]
  cublasHandle_t *handles = new cublasHandle_t[nthreads+1];// handles[]

  for(int i=0; i<nthreads+1; i++){
    checkCudaErrors(cudaStreamCreate(&streams[i]));
    checkCudaErrors(cublasCreate(&handles[i]));
  }

  unsigned int N = 40;
  vector<Task<double>> TaskList(N);
  initialize_tasks(TaskList);

  cout<<"Executing tasks on host / device"<<endl;

#ifdef USE_PTHREADS
  pthread_t threads[nthreads];
  threadData *InputToThreads = new threadData[nthreads];
  for(int i=0; i<nthreads; i++){
    checkCudaErrors(cudaSetDevice(dev_id));
    InputToThreads[i].tid = i;
    // TODO
    InputToThreads[i].streams = streams; // 的确是整个数组传进去的
    InputToThreads[i].handles = handles;
    // 如果刚好分配完
    if((TaskList.size()/nthreads) == 0){
      InputToThreads[i].taskSize = (TaskList.size()/nthreads);
      InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size()/nthreads)];
    }else{
      // 最后一个接收所有剩下的任务
      if(i == nthreads - 1){
        InputToThreads[i].taskSize = (TaskList.size() / nthreads) + (TaskList.size()%nthreads);
        InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size()/nthreads)+(TaskList.size()%nthreads)]
      }else{
        InputToThreads[i].taskSize = (TaskList.size()/nthreads);
        InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size()/nthreads)];
      }
    }
    pthread_create(&threads[i], NULL, &execute, &InputToThreads[i]);
  }

  for(int i=0; i<nthreads; i++)
    pthread_join(threads[i], NULL);
#else
  omp_set_num_threads(nthreads);
  #pragma omp parallel for schedule(dynamic)
  for(unsigned int i=0; i<TaskList.size(); i++){
    checkCudaErrors(cudaSetDevice(dev_id));
    int tid = omp_get_thread_num();
    execute(TaskList[i], handles, streams, tid);
  }
#endif

   checkCudaErrors(cudaDeviceSynchronize());
   for(int i=0; i<nthreads+1; i++){
      cudaStreamDestroy(streams[i]);
      cublasDestroy(handles[i]);
   }
   //free tasklist
   vector<Task<double>>().swap(TaskList);
   cout<<"All Done"<<endl;
  
}
