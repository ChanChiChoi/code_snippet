#include<iostream>
#include<cstdlib>

#include"Message.hpp"

using std::endl;
int
main(){
  Message hello{"Hello World Cmake"};
  std::cout<<hello<<endl;

  Message goodbye{"Goodbye World Cmake"};
  std::cout<<goodbye<<endl;

  return EXIT_SUCCESS;
}
