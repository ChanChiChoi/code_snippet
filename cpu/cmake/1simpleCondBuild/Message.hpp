#pragma once

#include<string>
#include<iosfwd>

class Message{
  public:
    Message(std::string const &m):message_{m}{}
    friend std::ostream &operator<<(std::ostream &os, Message &obj){
      return obj.print(os);
    }
  private:
    std::string message_;
    std::ostream &print(std::ostream &os);
};
