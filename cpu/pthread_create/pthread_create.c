#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<unistd.h>


struct member{
  int num;
  char *name;
};

static void*
pthread(void *arg){
  struct member *temp;
  printf("pthread start\n");
  sleep(2);

  temp = (struct member*) arg;
  printf("member->num:%d\n",temp->num);
  printf("member->name:%s\n",temp->name);
  return NULL;
}

int
main(int argc, char*argv[]){
  pthread_t tidp;
  struct member *b;

  b = (struct member*)malloc(sizeof(struct member));
  b->num = 1;
  b->name = "after";

  if((pthread_create(&tidp, NULL, pthread, (void*)b)) == -1){
    printf("create error\n");
    return 1;
  }
  sleep(1);

  printf("main continue\n");
  if(pthread_join(tidp, NULL)){
    printf("thread is not exit...\n");
    return -2;
  }
  return 0;
}

