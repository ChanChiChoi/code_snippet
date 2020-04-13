/*
 *
 * 障碍同步：在执行某个任务前，必须完成N个线程各自的任务，才能往下执行。主要有 pthread_barrier_wait 完成。
 *
 * pthread_barrier_wait：同步当前线程，使其在barrier对象处同步。当在该barrier处执行pthread_barrier_wait的线程数量达到预先设定值后，该线程会得到PTHREAD_BARRIER_SERIAL_THREAD返回值，其他线程得到0返回值。barrier对象会被reset到最近一次init的状态
 *
*/

#define _XOPEN_SOURCE 600

#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>

#include <pthread.h>
#include <stdio.h>
 
pthread_barrier_t b;
 
void* task(void* param) {
    int id = *(int*) param;
    printf("before the barrier %d\n", id);
    pthread_barrier_wait(&b);
    printf("after the barrier %d\n", id);
}
 
int main() {
    int nThread = 5;
    int i;
 
    pthread_t thread[nThread];
    pthread_barrier_init(&b, 0, nThread);

    for (i = 0; i < nThread; i++)
        pthread_create(&thread[i], 0, task, (void*)&i);

    for (i = 0; i < nThread; i++)
        pthread_join(thread[i], 0);

    pthread_barrier_destroy(&b);
    return 0;

}
