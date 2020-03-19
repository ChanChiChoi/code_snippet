#include<pthread.h>
#include<stdio.h>

void other_function(){
  pthread_exit(NULL);
}

void *entry_point(void *arg){
  printf("hello world\n");
  other_function();
  printf("can you see me?\n");
  return NULL;
}
int main(int argc, char *argv[]){
  pthread_t tidp;
  if(pthread_create(&tidp, NULL, &entry_point, NULL)){
    printf("could not create thread\n");
    return -1;
  }

  if(pthread_join(tidp, NULL)){
    printf("could not join thread\n");
    return -1;
  }
  printf("end\n");
  return 0;
}
