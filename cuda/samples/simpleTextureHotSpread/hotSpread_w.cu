#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<iostream>
#include<helper_cuda.h>

#define DIM 1024
#define PI 3.1415926f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

using namespace cv;
using namespace std;

texture<float> texConstSrc;
texture<float> texIn;
texture<float> texOut;

struct DataBlock{
  unsigned char *output_bitmap;
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;
  Mat *bitmap;

  cudaEvent_t st, ed;
  float totalTime;
  float frames;
};

//-------------------------------
__device__ unsigned char value(float n1, float n2, int hue){
  if(hue>360) hue -= 360;
  else if(hue<0) hue += 360;
  if(hue<60)
    return (unsigned char)(255*(n1+(n2-n1)*hue/60));
  if(hue<180)
    return (unsigned char)(255*n2);
  if(hue<240)
    return (unsigned char)(255*(n1+(n2-n1)*(240-hue)/60));
  return (unsigned char)(255*n1);
  
}

__global__ void
float_to_color(unsigned char *optr, float const *outSrc){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x+y*blockDim.x*gridDim.x;

  float l = outSrc[offset];
  float s = 1;
  int h = (180+(int)(360.f*outSrc[offset]))%360;
  float m1, m2;
  
  if(l<=0.5f)
    m2 = l*(1+s);
  else
    m2 = l+s-1*s;
  m1 = 2*l-m2;

  optr[offset*4+0] = value(m1,m2,h+120);
  optr[offset*4+1] = value(m1,m2,h);
  optr[offset*4+2] = value(m1,m2,h-120);
  optr[offset*4+3] = 255;
}
//--------------------------------
__global__ void
copy_const_kernel(float *iptr, float const *cptr=nullptr){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x+y*blockDim.x*gridDim.x;

  // cptr表示热源图，将其中非0的，也就是热源和热黑洞，覆盖到输出图上
  float c = tex1Dfetch(texConstSrc, offset);
  if(c != 0)
    iptr[offset] = c;
}

__global__ void
blend_kernel(float *dst, bool dstOut){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x+y*blockDim.x*gridDim.x;

  int left = offset - 1;
  int right = offset + 1;
  if(x==0) left++;
  if(x == DIM-1) right--;

  int top = offset - DIM;
  int bottom = offset + DIM;
  if(y == 0) top += DIM;
  if(y == DIM-1) bottom -= DIM;
  float t,l,c,r,b;
  if(dstOut){
    t = tex1Dfetch(texIn,top);
    l = tex1Dfetch(texIn,left);
    c = tex1Dfetch(texIn,offset);
    r = tex1Dfetch(texIn,right);
    b = tex1Dfetch(texIn,bottom);
  }else{
    t = tex1Dfetch(texOut,top);
    l = tex1Dfetch(texOut,left);
    c = tex1Dfetch(texOut,offset);
    r = tex1Dfetch(texOut,right);
    b = tex1Dfetch(texOut,bottom);
  }
  dst[offset] = c + SPEED*(t+b+r+l -4*c);
}
//------------------------------------

void anim_gpu(DataBlock *d, int ticks){
  checkCudaErrors(cudaEventRecord(d->st,0));
  dim3 grid(DIM/16, DIM/16);
  dim3 threads(16,16);
  Mat *bitmap = d->bitmap;

  volatile bool dstOut = true;
  for(int i=0; i<90; i++){
    float *in, *out;
    if(dstOut){
      in = d->dev_inSrc;
      out = d->dev_outSrc;
    }else{
      out = d->dev_inSrc;
      in = d->dev_outSrc;
    }
    copy_const_kernel<<<grid, threads>>>(in);
    blend_kernel<<<grid,threads>>>(out,dstOut);
    // 指针交换
    dstOut = !dstOut;
  }
  // 将每个浮点数转换成颜色值, 浮点数即像素点，表示当前值大小，
  // 然后需要将该像素值的大小(假设就是灰度值)，转换成颜色值 
  float_to_color<<<grid,threads>>>(d->output_bitmap, d->dev_inSrc);

  checkCudaErrors(cudaMemcpy(bitmap->data, d->output_bitmap, bitmap->elemSize()*bitmap->total(),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaEventRecord(d->ed,0));
  checkCudaErrors(cudaEventSynchronize(d->ed));
  float elapsedTime;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->st, d->ed));
  d->totalTime += elapsedTime;
  ++d->frames;
  cout<<"["<<d->frames<<"] frames total take times:"<<d->totalTime<<" ms;"
      <<"cur take time:"<<elapsedTime<<" ms"<<endl;
}


//===================================

int main(int argc, char *argv[]){
  DataBlock data;
  Mat bitmap(DIM,DIM,CV_8UC4, Scalar(0,0,0,0));
  data.bitmap = &bitmap;
  data.totalTime = 0;
  data.frames = 0;

  checkCudaErrors(cudaEventCreate(&data.st)); 
  checkCudaErrors(cudaEventCreate(&data.ed)); 
  checkCudaErrors(cudaEventRecord(data.st,0));
  checkCudaErrors(cudaMalloc((void**)&data.output_bitmap, bitmap.elemSize()*bitmap.total()));
  
  checkCudaErrors(cudaMalloc((void**)&data.dev_inSrc, bitmap.elemSize()*bitmap.total()));
  checkCudaErrors(cudaMalloc((void**)&data.dev_outSrc, bitmap.elemSize()*bitmap.total()));
  checkCudaErrors(cudaMalloc((void**)&data.dev_constSrc, bitmap.elemSize()*bitmap.total()));
  // 纹理绑定
  checkCudaErrors(cudaBindTexture(NULL, texConstSrc, data.dev_constSrc,bitmap.elemSize()*bitmap.total()));
  checkCudaErrors(cudaBindTexture(NULL, texIn, data.dev_inSrc,bitmap.elemSize()*bitmap.total()));
  checkCudaErrors(cudaBindTexture(NULL, texOut, data.dev_outSrc,bitmap.elemSize()*bitmap.total()));
 
  // 随机生成热源点,这里用float代表一个像素点的rgba四个值
  float *temp = (float*)malloc(bitmap.elemSize()*bitmap.total());

  for(int i=0; i<DIM*DIM; i++){
    temp[i] = 0.f;
    int x = i%DIM;
    int y = i/DIM;
    // 在[xmin,ymin,xmax,ymax]=[300,310,600,601]区域是个白热区
    if((x>300) && (x<600) && (y>310) && (y<601))
      temp[i] = MAX_TEMP;
  }
  // 图中间加个半热源
  temp[DIM*100 + 100] = (MAX_TEMP+MIN_TEMP)/2;
  // 几个地方为热度黑洞，即这个区域热度永远为0
  temp[DIM*700 + 100] = MIN_TEMP;
  temp[DIM*300+300] = MIN_TEMP;
  temp[DIM*200+700] = MIN_TEMP;
  for(int y=800; y<900; y++){
    for(int x=400; x<500; x++){
      temp[x+y*DIM] = MIN_TEMP;
    }
  }
  // 将初始化好的作为const图，
  checkCudaErrors(cudaMemcpy(data.dev_constSrc, temp, bitmap.elemSize()*bitmap.total(),
                             cudaMemcpyHostToDevice));
  //------------------------
  //新增高温区域,该区域会越来越小，证明热量在散失
  for(int y=800; y<DIM; y++){
    for(int x=0; x<200; x++){
      temp[x+y*DIM] = MAX_TEMP;
    }
  }
  checkCudaErrors(cudaMemcpy(data.dev_inSrc, temp, bitmap.elemSize()*bitmap.total(),
                             cudaMemcpyHostToDevice));
  free(temp);
  for(int i=0; i<190; i++){
    anim_gpu(&data,1);
//    imshow("display",*data.bitmap);
//    waitKey(1);
  }

  checkCudaErrors(cudaEventDestroy(data.st));
  checkCudaErrors(cudaEventDestroy(data.ed));
  checkCudaErrors(cudaUnbindTexture(texIn));
  checkCudaErrors(cudaUnbindTexture(texOut));
  checkCudaErrors(cudaUnbindTexture(texConstSrc));
  checkCudaErrors(cudaFree(data.output_bitmap));
  checkCudaErrors(cudaFree(data.dev_inSrc));
  checkCudaErrors(cudaFree(data.dev_outSrc));
  checkCudaErrors(cudaFree(data.dev_constSrc));
  
}
