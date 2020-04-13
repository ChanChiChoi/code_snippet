#include"multithreading.h"


//创建线程
CUTThread cutStartThread(CUT_THREADROUTING func,void *data){
  pthread_t threads;
  pthread_create(&thread, NULL, func, data);
  return threads;
}

//等待线程结束
void cutEndThread(CUTThread thread){
  pthread_join(thread, NULL);
}

//阻塞多个线程
void cutWaitForThreads(CUTThread const* threads, int num){
  for(int i=0; i<num; i++)
    cutEndThread(threads[i]);
}

//创建barrier
CUTBarrier cutCreateBarrier(int releaseCount){
  CUTBarrier barrier;
  barrier.count = 0;
  barrier.releaseCount = releaseCount;
  pthread_mutex_init(&barrier.mutex,0);
  pthread_cond_init(&barrier.conditionVariable, 0);
  return barrier;
}

//Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier * barrier){
  int myBarrierCount;
  pthread_mutex_lock(&barrier->mutex);
  myBarrierCount = ++barrier->count;
  pthread_mutex_unlock(&barrier->mutex);

  if(myBarrierCount >= barrier->releaseCount)
    pthread_cond_signal(&barrier->conditionVariable);
}

//等Barrier释放
void cutWaitForBarrier(CUTBarrier* barrier){

  pthread_mutex_lock(&barrier->mutex);
  while(barrier->count < barrier->releaseCount)
    pthread_cond_wait(&barrier->conditionVariable,
                      &barrier->mutex);

  pthread_mutex_unlock(&barrier->mutex);
}

//销毁barrier
void cutDestroyBarrier(CUTBarrier* barrier){
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->conditionVariable);
}

