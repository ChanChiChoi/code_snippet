#include"config.h"
#include<cstdlib>
#include<iostream>

using std::cout;
using std::endl;
int
main(){
  cout<<"逻辑核个数:"<<NUMBER_OF_LOGICAL_CORES<<endl;
  cout<<"物理核个数:"<<NUMBER_OF_PHYSICAL_CORES<<endl;
  cout<<"总的虚拟内存(MB):"<<TOTAL_VIRTUAL_MEMORY<<endl;
  cout<<"可用虚拟内存(MB):"<<AVAILABLE_VIRTUAL_MEMORY<<endl;
  cout<<"总的物理内存(MB):"<<TOTAL_PHYSICAL_MEMORY<<endl;
  cout<<"可用物理内存(MB):"<<AVAILABLE_PHYSICAL_MEMORY<<endl;
  cout<<"处理器是否是64位:"<<IS_64BIT<<endl;
  cout<<"处理器是否有浮点单元(FPU):"<<HAS_FPU<<endl;
  cout<<"处理器是否支持MMX指令集:"<<HAS_MMX<<endl;
  cout<<"处理器是否支持Ext. MMX指令集:"<<HAS_MMX_PLUS<<endl;
  cout<<"处理器是否支持SSE指令集:"<<HAS_SSE<<endl;
  cout<<"处理器是否支持SSE2指令集:"<<HAS_SSE2<<endl;
  cout<<"处理器是否支持SSE FP指令集:"<<HAS_SSE_FP<<endl;
  cout<<"处理器是否支持SSE MMX指令集:"<<HAS_SSE_MMX<<endl;
  cout<<"处理器是否支持3DNow指令集:"<<HAS_AMD_3DNOW<<endl;
  cout<<"处理器是否支持3DNow+指令集:"<<HAS_AMD_3DNOW_PLUS<<endl;
  cout<<"IA64处理器模拟x86:"<<HAS_IA64<<endl;
  cout<<"操作系统名称:"<<OS_NAME<<endl;
  cout<<"操作系统子类型:"<<OS_RELEASE<<endl;
  cout<<"操作系统构建ID:"<<OS_VERSION<<endl;
  cout<<"操作系统平台:"<<OS_PLATFORM<<endl;
  return EXIT_SUCCESS;
}
