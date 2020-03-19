#include<sys/types.h>
#include<sys/syscall.h>
#include<unistd.h>
#include<pthread.h>
#include<stdio.h>
#include<math.h>

#define INTERATIONS 30

typedef unsigned  long lu;

pthread_mutex_t mutex;
double target;

pid_t getpid(void);
pid_t gettid(void){
  return syscall(SYS_gettid);
}

void *opponent(void *arg){
  int i;
  for( i=0; i<INTERATIONS; i++){
    pthread_mutex_lock(&mutex);
    printf("tid=[%lu], pid=%lu get mutex\n", (lu)gettid(),(lu)getpid());
    sleep(0.5);
    pthread_mutex_unlock(&mutex);
  }
  return NULL;
}
int main(int argc, char*argv[]){

  pthread_t tidp;
  if(pthread_mutex_init(&mutex,NULL)){
    printf("unable to initialize a mutex\n");
    return -1;
  }

  if(pthread_create(&tidp,NULL,&opponent,NULL)){
    printf("unable to spwan thread\n");
    return -1;
  }
  int i;
  for( i=0; i<INTERATIONS; ++i){
    pthread_mutex_lock(&mutex);
    printf("tid=[%lu], pid=%lu get mutex\n", (lu)gettid(),(lu)getpid());
    sleep(0.5);
    pthread_mutex_unlock(&mutex);
  }
  if(pthread_join(tidp,NULL)){
    printf("could not join thread\n");
    return -1;
  }

  pthread_mutex_destroy(&mutex);
  printf("end\n");
  return 0;
}
