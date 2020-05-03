#include"Message.hpp"

#include<iostream>
#include<string>


std::ostream &Message::print(std::ostream &os){
  os<<"This is my nice message: "<<std::endl;
  os<<message_;
  return os;
}
