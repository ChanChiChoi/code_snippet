/*
sem_init: Initialize a new semaphore. Note, the second argument denotes how the semaphore will be shared. Passing zero denotes that it will be shared among threads rather than processes. The final argument is the initial value of the semaphore.
sem_destroy: Deallocate an existing semaphore.
sem_wait: This is the P() operation.
sem_post: This is the V() operation.

 * */
#include<semaphore.h>
#include<pthread.h>
#include<stdio.h>

#define THREADS 20

sem_t OkToBuyThing;
int ThingAvaliable;

void *
buyer(void *arg){
 //P()
 sem_wait(&OkToBuyThing);
 if(!ThingAvaliable){
  // buy some thing;
  ++ThingAvaliable;
 }
 //V()
 sem_post(&OkToBuyThing);
 return NULL;
}

int main(int argc, char*argv[]){
  pthread_t tidp[THREADS];
  ThingAvaliable = 0;
  
  //initialize the semaphore with 1.
  // the second argument passing 0 denotes
  // that the semaphore is shared between threads and
  // not processes;
  if(sem_init(&OkToBuyThing, 0, 1)){
    printf("could not initialize a semaphore\n");
    return -1;
  }
 
  int i=0;
  for(;i<THREADS;i++){
    if(pthread_create(&tidp[i],NULL,&buyer,NULL)){
      printf("could not create thread %d\n",i);
      return -1;
    }

  }

  for(i=0;i<THREADS;++i){
    if(pthread_join(tidp[i],NULL)){
      printf("could not join thread %d\n",i);
      return -1;
    }

  }
  sem_destroy(&OkToBuyThing);

  printf("total thing %d\n",ThingAvaliable);
  return 0;
}
