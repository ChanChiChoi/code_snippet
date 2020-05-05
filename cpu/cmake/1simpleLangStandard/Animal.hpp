#pragma once

#include<string>

using std::string;

class Animal{
 public:
   Animal() = delete;
   explicit Animal(string const &n);
   string name() const;
 protected:
   string name_;
 private:
   virtual string name_impl() const = 0;
};
