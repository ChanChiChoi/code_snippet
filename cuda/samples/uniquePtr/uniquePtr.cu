#include<cstdio>
#include<memory>
#include<iostream>

using namespace std;

__global__ void
test(int *d_data){

  printf("hello world\n");
  for(int i = 0;i<10;i++)
    printf("%d:%d\n",i,d_data[i]);
}

int main(){


 unique_ptr<int,void(*)(int *)> d_data{nullptr,[](int *p){cudaFree(p);}}; 

 cudaMalloc((void**)&(d_data.get()),sizeof(int)*10);

 int h_data[10] = {1,2,3,4,5,6,7,8,9,10};

 cudaMemcpy(d_data,h_data,sizeof(int)*10,cudaMemcpyHostToDevice);
 test<<<1,1>>>(d_data);


}
