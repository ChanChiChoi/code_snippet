#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>

#include<iostream>

#include<cuda_gl_interop.h>
#include<helper_cuda.h>

#define DIM 512

GLuint bufferObj;
cudaGraphicsResource *resource;


int
main(int argc, char *argv[]){
  cudaDeviceProp prop;
  int dev;
  memset(&prop,0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  checkCudaErrors(cudaChooseDevice(&dev, &prop));

  checkCudaErrors(cudaGLSetGLDevice(dev));

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
  glutInitWindowSize(DIM,DIM);
  glutCreateWindow("bitmap");
}


