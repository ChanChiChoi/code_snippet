#include"Animal.hpp"

#include<string>

using std::string;

Animal::Animal(string const &n):name_{n}{}

string Animal::name() const{return name_impl();}

