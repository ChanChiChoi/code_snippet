#ifndef MULTITHREADING_H
#define MULTITHREADING_H


#include<pthread.h>

using CUTThread = pthread_t;
using CUT_THREADROUTING = void*(*)(void);

#define CUT_THREADPROC void*
#define CUT_THREADEND return 0

struct CUTBarrier{
  pthread_mutex_t mutex;
  pthread_cond_t conditionVariable;
  int releaseCount;
  int count;
};


#ifdef __cplusplus
extern "C"{
#endif


//创建线程
CUTThread cutStartThread(CUT_THREADROUTING,void *data);

//等待线程结束
void cutEndThread(CUTThread thread);

//阻塞多个线程
void cutWaitForThreads(CUTThread const* threads, int num);

//创建barrier
CUTBarrier cutCreateBarrier(int releaseCount);

//Increment barrier. (execution continues)
void cutIncrementBarrier(CUTBarrier * barrier);

//等Barrier释放
void cutWaitForBarrier(CUTBarrier* barrier);

//销毁barrier
void cutDestroyBarrier(CUTBarrier* barrier);

#ifdef __cplusplus
}
#endif

#endif
