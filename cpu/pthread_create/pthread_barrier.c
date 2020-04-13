/*
==========================================
This program spawns a number of threads, assigning each to compute part of a matrix multiplication. Each thread then uses the result of that computation in the next phase: another matrix multiplication.

There are a few things to note here:

1-The barrier declaration at the top
2-The barrier initialization in main
3-The point where each thread waits for its peers to finish.

NOTE:
The preprocessor definition of _XOPEN_SOURCE at the top of the program is important; without it, the barrier prototypes are not defined in pthread.h. The definition must come before any headers are included.

====================================
*/

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

#define ROWS 10000
#define COLS 10000
#define THREADS 10

double initial_matrix[ROWS][COLS];
double final_matrix[ROWS][COLS];

pthread_barrier_t barr;

extern void
DotProduct(int row, int col,
           double src[ROWS][COLS],
           double dst[ROWS][COLS]);

extern double
Determinant(double matrix[ROWS][COLS]);

void *entry_point(void *arg){

  int rank = *(int*)arg;
  for(int row = rank*ROWS/THREADS; row<(rank+1)*THREADS; ++row)
    for(int col=0; col<COLS;++col)

      DotProduct(row,col,initial_matrix,final_matrix);

  //Synchronization point
  int rc = pthread_barrier_wait(&barr);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD){
    printf("could not wait on barrier\n");
    exit(-1);
  }

  for(int row = rank*ROWS/THREADS; row<(rank+1)*THREADS; ++row)
    for(int col=0;col<COLS;++col)
     DotProduct(row,col,final_matrix, initial_matrix);
}

int main(int argc, char*argv[]){
  pthread_t tidp[THREADS];
  
  //barrier init
  if(pthread_barrier_init(&barr,NULL,THREADS)){
    printf("could not create a barrier\n");
    return -1;
  }
  for(int i=0; i<THREADS; ++i)
    if(pthread_create(&tidp[i],NULL,&entry_point, (void*)i)){
      printf("could not create thread %d\n",i);
      return -1;
    }

  for(int i=0; i<THREADS;++i)
    if(pthread_join(tidp[i],NULL)){
      printf("could not join thread %d\n",i);
      return -1;
    }

  double det = Determinant(initial_matrix);
  printf("the determinant of M^4 = %f\n",det);

  return 0;
}
