#include<stdio.h>
#include<stdlib.h>
#include<dlfcn.h>

int main(int argc, char **argv){
  void *handle;
//  void (*callfun)();
  double (*round)();
  char *error;
  handle = dlopen("/lib64/libm-2.17.so",RTLD_LAZY);

  if (!handle){
    printf("%s \n",dlerror());
    exit(1);
  }

  dlerror();

  round = dlsym(handle,"round");
  if((error = dlerror()) != NULL){
    printf("%s \n",error);
    exit(1);
  }
  double ans = round(1.2);
  printf("round 1.2 = %f\n",ans);
  dlclose(handle);
  return 0;
  
}
