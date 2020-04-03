#include<cstdio>
#include<memory>
#include<vector>
#include<functional>
#include<iostream>

using namespace std;
using fp = void(*)(int*);


__global__ void
test(int *d_data){

  printf("hello world\n");
  for(int i = 0;i<10;i++)
    printf("%d:%d\n",i,d_data[i]);
}


int uniquePtr(){

 cout<<"uniquePtr"<<endl;
 int *d_data0;
 function<void(int*)> lambda = [](int*p){cudaFree(p);};  
 unique_ptr<int,function<void(int*)>> d_data{nullptr, lambda}; 

 cudaMalloc((void**)&d_data0,sizeof(int)*10);
 d_data.reset(d_data0);
 //交给unique_ptr做指针维护
 // unique_ptr 的生命周期要与cudaDeviceReset一起考虑，
 //cudaDeviceReset是将上下文都重置，如果之前并未执行cudaFree则会造成内存泄漏
// 但是，如果不调用cudaDeviceReset,其会在main函数生命周期之后执行
 int h_data[10] = {1,2,3,4,5,6,7,8,9,10};

 cudaMemcpy(d_data.get(),h_data,sizeof(int)*10,cudaMemcpyHostToDevice);
 test<<<1,1>>>(d_data.get());
 cudaDeviceSynchronize();
 return 0;
}


int uniquePtr1(){

  cout<<"uniquePtr1"<<endl;
  function<void(int*)> lambda = [](int*p){cudaFree(p);};  
  vector<unique_ptr<int,function<void(int*)>> > vec;

  for(int i=0; i<2;i++){
    vec.emplace_back(nullptr,lambda);
    int* tmp;
    cudaMalloc((void**)&tmp,sizeof(int)*10);
    vec[i].reset(tmp);

    int h_data[10] = {1,2,3,4,5,6,7,8,9,10};
    cudaMemcpy(vec[i].get(),h_data,sizeof(int)*10,cudaMemcpyHostToDevice);
    test<<<1,1>>>(vec[i].get());
  }
  
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
  cout<<"-------------"<<endl;
  uniquePtr1();
#else
  normal();
#endif
  //一定要加上这句，不然底层context会自己帮忙释放未释放的内存，
  //显示调用就意味着内存需要手动自己释放
  cudaDeviceReset();
  return 0;

}
