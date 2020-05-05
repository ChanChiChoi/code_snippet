#pragma once

#include<cstdio>
#include<cstdlib>
#include<map>
#include<sstream>
#include<string>
#include<type_traits>

#define ERROR(mes) {\
    std::ostringstream _err; \
    _err <<"Fatal error.\n"  \
         <<" In function "<<__func__<<" at line "<<__LINE__<<" of file" \
         <<mes<<std::endl;    \
     std::fprintf(stderr, "%s\n", _err.str().c_str()); \
    std::exit(EXIT_FAILURE);  \
 }

namespace detail{
  template<typename CreateObject> 
  class BaseFactory{
    private:
      typedef std::map<std::string, CreateObject> CallbackMap;
      typedef typename CallbackMap::value_type CallbackPair;
      typedef typename CallbackMap::const_iterator CallbackConstIter;
    protected:
      CallbackMap callbacks_;
      CallbackConstIter retrieve(std::string const &objID) const{
         if(objID.empty())
           ERROR("No object identification string provided to the Factory");
        
         CallbackConstIter i = callbacks_.find(objID);
         if(i == callbacks_.end())
           ERROR("The unknown object ID "+objID+" occurrend in the Factory.");
         return i;
      }
    private:
      bool registerObject(std::string const &objID,
                          const CreateObject &functor){
        return callbacks_.insert(CallbackPair(objID,functor)).second;
      }
      bool unRegisterObject(std::string const &objID){
        return callbacks_.erase(objID) == 1;
      }
    public:
      void subscribe(std::string const &objID, 
                     CreateObject const &functor){
        bool done = this->registerObject(objID, functor);
        if(!done)
           ERROR("Subscription of object ID "+objID+" occurrend in the Factory.");
      }
      void unsubscribe(std::string const &objID){
        bool done = this->unRegisterObject(objID);
        if(!done)
           ERROR("Unsubscription of object ID "+objID+" occurrend in the Factory.");
      }
  };
}

template<typename CreateObject>
class Factory final: public detail::BaseFactory<CreateObject>{
  public:
    template<typename... ObjectInputArgs>
    typename std::result_of<CreateObject(ObjectInputArgs...)>::type 
    create(std::string const &objID,
           ObjectInputArgs... data) const{
       return (this->retrieve(objID)->second)(data...);
    }
};
