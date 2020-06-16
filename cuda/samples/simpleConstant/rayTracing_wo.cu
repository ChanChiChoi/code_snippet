#include<iostream>
#include<helper_cuda.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20
#define DIM  2000

#define INF 2e10f
using cv::imwrite;
using cv::imshow;
using cv::Mat;
using cv::waitKey;
using cv::Scalar;
using std::endl;
using std::cout;

// 相机固定在Z轴，面向原点。此外不支持其他照明，也不计算照明效果
// 每个球面分配一个颜色值，而且是可见的
// 因为没照明，所以就是平行光从z轴负方向区域射向相机。
struct Sphere{
  float r,g,b;
  float radius;
  float x,y,z;
  // 相机在z轴，则二维图片在xy-z成像，所以计算二维图片每个像素点的光线来源
  // 光线来自[ox,oy]处的像素点，首先计算是否相交，
  // 然后计算得出相机到光线命中球面处的距离
  __device__ float hit(float ox, float oy, float *n){
    float dx = ox-x; // 
    float dy = oy-y;
    // 基于xy面进行投影，计算当前像素点在xy面上是否投影到求切面内
    if(dx*dx + dy*dy < radius*radius){
      float dz = sqrtf(radius*radius - dx*dx - dy*dy);  // 球心为相对原点，计算像素的高
      *n = dz / sqrtf(radius*radius);
      return dz + z;
    }
    return -INF;
  }
};

Sphere *d_s;

__global__ void
kernel(unsigned char *ptr,Sphere *d_s){
  //各自处理一个像素点
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int offset = x+y*blockDim.x*gridDim.x;
  float ox = (x - DIM/2);
  float oy = (y - DIM/2);

  float r=0,g=0,b=0;
  float maxz = -INF;
  // 每个像素点都需要计算20个球
  for(int i=0; i<SPHERES; i++){
    float n;
    float t = d_s[i].hit(ox,oy,&n);
    if(t>maxz){
      float fscale = n;
      r = d_s[i].r*fscale;
      g = d_s[i].g*fscale;
      b = d_s[i].b*fscale;
      maxz = n;// 应该加
    }
  }

  ptr[offset*4+0] = (int)(r*255);
  ptr[offset*4+1] = (int)(g*255);
  ptr[offset*4+2] = (int)(b*255);
  ptr[offset*4+3] = 255;
}

int main(int argc, char *argv[]){
  cudaEvent_t st, ed;
  checkCudaErrors(cudaEventCreate(&st));
  checkCudaErrors(cudaEventCreate(&ed));
  checkCudaErrors(cudaEventRecord(st,0));

 // Bitmap bitmap(DIM, DIM);
  Mat bitmap(DIM, DIM, CV_8UC4, Scalar(0,0,0,0));
  unsigned char *d_bitmap;
  checkCudaErrors(cudaMalloc((void**)&d_bitmap, bitmap.total()*4));
  checkCudaErrors(cudaMalloc((void**)&d_s,sizeof(Sphere)*SPHERES));
  // cpu侧构建20个球的存储空间
  Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
  for(int i=0; i<SPHERES; i++){
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f)-500;
    temp_s[i].y = rnd(1000.0f)-500;
    temp_s[i].z = rnd(1000.0f)-500;
    temp_s[i].radius = rnd(100.0f)+20;
  }

  checkCudaErrors(cudaMemcpy(d_s,temp_s,sizeof(Sphere)*SPHERES,
                             cudaMemcpyHostToDevice));
  free(temp_s);

  dim3 grid(DIM/16,DIM/16);
  dim3 threads(16,16);
  kernel<<<grid,threads>>>(d_bitmap,d_s);
  checkCudaErrors(cudaMemcpy(bitmap.data, d_bitmap, bitmap.total()*4,
                             cudaMemcpyDeviceToHost));
  imshow("test",bitmap);
  waitKey(0);
  //imwrite("1.png",bitmap);
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  float elasedTime;
  checkCudaErrors(cudaEventElapsedTime(&elasedTime,st,ed));
  cout<<"take time:"<<elasedTime<<" ms"<<std::endl;
  checkCudaErrors(cudaEventDestroy(st));
  checkCudaErrors(cudaEventDestroy(ed));
  cudaFree(d_bitmap);
  cudaFree(d_s);
  
}
