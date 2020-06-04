//vector_types 在cuda,提供int2
#include<vector_types.h>
#include<memory>


extern "C"
void computeGold(std::unique_ptr<char[]>&ref, char *idata, unsigned int const len);
extern "C"
void computeGold2(std::unique_ptr<int2[]>&ref, int2 *idata, unsigned int const len);

void 
computeGold(std::unique_ptr<char[]>&ref, char *idata, unsigned int const len){
  for(unsigned int i=0; i<len; i++){
    ref[i] = idata[i] - 10;
  }
}

void
computeGold2(std::unique_ptr<int2[]>&ref, int2 *idata, unsigned int const len){
  for(unsigned int i = 0; i<len; i++){
    ref[i].x = idata[i].x - idata[i].y;
    ref[i].y = idata[i].y;
  }
}
