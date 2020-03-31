#include<iostream>
#include<stdio.h>
#include<helper_cuda.h>
#include<memory>

using namespace std;

#define MAX_DEPTH 16
#define INSERTION_SORT 32



void initialize_data(unsigned int *dst, unsigned int nitems){
  srand(2047);
  for(unsigned i=0; i<nitems; i++)
    dst[i] = rand()%nitems;
}

void check_results(int n, unsigned int *results_d){

  unique_ptr<unsigned int[]> results_h{new unsigned int[n]};
  checkCudaErrors(cudaMemcpy((void**)(results_h.get()),
                             results_d,n*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  //就检查是否前一个值大于当前值，就知道是否有序了
  for(int i=1; i<n; ++i){
    if(results_h[i-1]>results_h[i]){
      cerr<<" Invalid item["<<i-1<<"]:"<<results_h[i-1]<<" bigger than "<<results_h[i]<<endl;
      exit(EXIT_FAILURE);
    }
  }
  cout<<"OK"<<endl;
}


__global__ void cdp_simple_quicksort(unsigned int *data, int, int, int);

void run_qsort(unsigned int *data, unsigned int nitems){
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,MAX_DEPTH));

  int left = 0;
  int right = nitems-1;
  cout<<" Launching kernel on the GPU"<<endl;
  cdp_simple_quicksort<<<1,1>>>(data,left,right,0);
  checkCudaErrors(cudaDeviceSynchronize());

}

__device__ void
selection_sort(unsigned int *data, int left, int right){
  for(int i=left; i<= right; ++i){

    unsigned int min_val = data[i];
    int min_idx = i;

    //获取最小值
    for(int j=i+1; j<=right; ++j){
      unsigned int val_j = data[j];
      // 保存最小值
      if(val_j < min_val){
        min_idx = j;
        min_val = val_j;
      }
    }
    //交换
    if(i!= min_idx){
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void
cdp_simple_quicksort(unsigned int *data, int left, int right, int depth){
  //如果太深，或者只剩下很少的元素需要排序，就启用插入排序方法
  if(depth >= MAX_DEPTH || right-left <= INSERTION_SORT){
     selection_sort(data, left, right); 
     return ;
  }
  
  unsigned int *lptr = data+left;
  unsigned int *rptr = data+right;
  unsigned int pivot = data[(left+right)/2];

  //保证中间值左边的小于中间值，右边的大于中间值
  while(lptr <= rptr){
    unsigned int lval = *lptr;
    unsigned int rval = *rptr;

    //找到左边大于中间值的
    while(lval<pivot){
      lptr++;
      lval = *lptr;
    }
    //找到右边小于中间值的
    while(rval>pivot){
      rptr--;
      rval = *rptr;
    }
    //经过上面查找后，
    //如果左指针小于右指针，说明存在中间点左边的值大于右边
    //否则就是左指针跑到右指针右边，那就无需基于中间点交换
    if(lptr<=rptr){
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  int nright = rptr - data;
  int nleft = lptr - data;
  //===============cpu的递归，这里采用新的流去做处理，
  // 快排左边的部分
  //如果当前左边起点left,还小于rptr当前位置，即左边部分还是存在无序的情况
  //即使左边的比中间值小
  if(left<nright){
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1,1,0,s>>>(data,left,nright,depth+1);
    cudaStreamDestroy(s);
  }

  //快排右边部分
  if(nleft<right){
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1,1,0,s1>>>(data,nleft,right,depth+1);
    cudaStreamDestroy(s1);
  }
}


int main(int argc, char *argv[]){

  int num_items = 128;

  //create input data
  unsigned int *h_data = 0;
  unsigned int *d_data = 0;

  cout<<"Initializing data:"<<endl;
  h_data = static_cast<unsigned int *>(malloc(num_items*sizeof(unsigned int)));
  initialize_data(h_data,num_items);

  //allocate gpu mem
  checkCudaErrors(cudaMalloc((void**)&d_data, num_items*sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_data, h_data, num_items*sizeof(unsigned int), cudaMemcpyHostToDevice));

  //execute
  cout<<"Running quicksort on "<<num_items<<" elements"<<endl;
  run_qsort(d_data, num_items);

  //check result
  cout<<"validating results: ";
  check_results(num_items, d_data);

  free(h_data);
  checkCudaErrors(cudaFree(d_data));

  exit(EXIT_SUCCESS);

}
