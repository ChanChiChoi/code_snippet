#pragma once
#include"Animal.hpp"
using std::string;

class Dog final: public Animal{
  public:
    Dog(string const &n):Animal{n}{}

  private:
    string name_impl() const override;
};
