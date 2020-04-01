#include<iostream>
#include<cuda_runtime.h>
#include<vector_types.h>
#include<helper_cuda.h>

using namespace std;

extern "C" bool
runTest(int const argc, char const * argv[],
        char *data, int2* data_int2, unsigned int len);

int
main(int argc, char * argv[]){

  int len = 16;
  char str[] = {82, 111,118,118,
                121,42, 97, 121,
                124,118,110,56,
                10, 10, 10, 10};

  int2 i2[16];

  for(int i=0; i<len; i++){
    i2[i].x = str[i];
    i2[i].y = 10;
  }
  bool bTestResult;

  bTestResult = runTest(argc, (char const **)argv, str, i2, len);

  cout<<str<<endl;
  exit(bTestResult? EXIT_SUCCESS: EXIT_FAILURE);
}
