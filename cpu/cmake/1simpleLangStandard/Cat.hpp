#pragma once

#include "Animal.hpp"

using std::string;

class Cat final: public Animal{
  public:
    Cat(string const &n):Animal{n}{}
  private:
    string name_impl() const override;
};
