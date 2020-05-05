#include"Dog.hpp"
using std::string;

string 
Dog::name_impl() const{
  return "I'm " + name_ + " the dog!";
}
