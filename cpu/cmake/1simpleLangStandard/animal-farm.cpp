#include"Animal.hpp"
#include"Cat.hpp"
#include"Dog.hpp"
#include"Factory.hpp"

#include<cstdlib>
#include<functional>
#include<iostream>
#include<memory>

typedef std::function<std::unique_ptr<Animal>(std::string const &)> CreateAnimal;

using std::string;
using std::make_unique;
using std::unique_ptr;
using std::cout;
using std::endl;

int
main(){
  Factory<CreateAnimal> farm;
  farm.subscribe("CAT",
                 [](string const &n){return make_unique<Cat>(n);});
  farm.subscribe("DOG",
                 [](string const &n){return make_unique<Dog>(n);});

  unique_ptr<Animal> simon = farm.create("CAT","Simon");
  unique_ptr<Animal> marlowe = farm.create("DOG","Marlowe");
  cout<<simon->name()<<endl;
  cout<<marlowe->name()<<endl;

  return 0;
  
}
