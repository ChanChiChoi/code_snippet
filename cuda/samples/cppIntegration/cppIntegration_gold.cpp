//vector_types 在cuda,提供int2
#include<vector_types.h>


extern "C"
void computeGold(char *ref, char *idata, unsigned int const len);
extern "C"
void computeGold2(int2 *ref, int2 *idata, unsigned int const len);

void 
computeGold(char *ref, char *idata, unsigned int const len){
  for(unsigned int i=0; i<len; i++){
    ref[i] = idata[i] - 10;
  }
}

void
computeGold2(int2 *ref, int2 *idata, unsigned int const len){
  for(unsigned int i = 0; i<len; i++){
    ref[i].x = idata[i].x - idata[i].y;
    ref[i].y = idata[i].y;
  }
}
