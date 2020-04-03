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

int uniquePtr(){

 cout<<"uniquePtr"<<endl;
 int *d_data0;
 cudaMalloc((void**)&d_data0,sizeof(int)*10);
 //交给unique_ptr做指针维护
 unique_ptr<int,void(*)(int *)> d_data{d_data0,
                                       [](int *p){cudaFree(p);} }; 
 int h_data[10] = {1,2,3,4,5,6,7,8,9,10};

 cudaMemcpy(d_data.get(),h_data,sizeof(int)*10,cudaMemcpyHostToDevice);
 test<<<1,1>>>(d_data.get());
 cudaDeviceSynchronize();
 return 0;

}

int normal(){
 
 cout<<"normal"<<endl;
 int *d_data;
 // 故意缺少cudaFree，调用cuda-memcheck
 cudaMalloc((void**)&d_data,sizeof(int)*10);
 int h_data[10] = {1,2,3,4,5,6,7,8,9,10};

 cudaMemcpy(d_data,h_data,sizeof(int)*10,cudaMemcpyHostToDevice);
 test<<<1,1>>>(d_data);
 cudaDeviceSynchronize();
 return 0;
}

int main(){

#ifdef UNIQUE
  uniquePtr();
#else
  normal();
#endif
  //一定要加上这句，不然底层context会自己帮忙释放未释放的内存，
  //显示调用就意味着内存需要手动自己释放
  cudaDeviceReset();
  return 0;

}
