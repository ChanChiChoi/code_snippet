#include<cstdlib>
#include<iostream>
#include<string>

using std::string;
using std::cout;
using std::endl;

string hello(){
#ifdef IS_WINDOWS
  return string{"Hello from Windows"};
#elif IS_LINUX
  return string{"Hello from Linux!"};
#elif IS_MACOS
  return string{"Hello from macOS"};
#else
  return string{"Hello from an unknown system!"};
#endif
}

string hello1(){
#ifdef IS_INTEL_CXX_COMPILER
  return string{"Hello Intel compiler"};
#elif IS_GNU_CXX_COMPILER
  return string{"Hello GNU compiler"};
#else
  return string{"Hello unknown compiler?"};
#endif
}

//即GCC在宏预处理阶段，有一条排外规则，
//那就是若宏参数被用于字符串化或者与其它标签连接，则不会被替代！
#define MAC(x) #x //将宏的参数进行字符串化
#define MAC_WARP(x) MAC(x) 
int
main(){
  cout<<hello()<<endl;
  cout<<hello1()<<endl;
#ifdef COMPILER_NAME
  cout<<"compilername:" MAC_WARP(COMPILER_NAME)<<endl;
#endif
  return EXIT_SUCCESS;
}
