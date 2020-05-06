#include<cstdlib>
#include<iostream>
#include<string>

using std::string;
using std::cout;
using std::endl;

//即GCC在宏预处理阶段，有一条排外规则，
//那就是若宏参数被用于字符串化或者与其它标签连接，则不会被替代！
#define STRINGIFY(x) #x //将宏的参数进行字符串化
#define TOSTRING(x) STRINGIFY(x) 

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

string hello2(){
  string arch_info(TOSTRING(ARCHITECTURE));
  arch_info += string{" architecture. "};
#ifdef IS_32_BIT_ARCH
  return arch_info + string{"Compiled on a 32 bit host processor."};
#elif IS_64_BIT_ARCH
  return arch_info + string{"Compiled on a 64 bit host processor."};
#else
  return arch_info + string{"Neither 32 or 64 bit host processor."};
#endif
}

int
main(){
  cout<<hello()<<endl;
  cout<<hello1()<<endl;
  cout<<hello2()<<endl;
#ifdef COMPILER_NAME
  cout<<"compilername:" TOSTRING(COMPILER_NAME)<<endl;
#endif
  return EXIT_SUCCESS;
}
