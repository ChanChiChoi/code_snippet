#include "Cat.hpp"

using std::string;
string 
Cat::name_impl() const{
  return "I'm " + name_ + " the cat!";
}
