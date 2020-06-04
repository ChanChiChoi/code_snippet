#include<cmath>
#include<cstdio>
#include<iostream>


using namespace std;


extern "C"
int computeGold(int *, int const);


int
computeGold(int *gpuData, int const len){

  //add
  int val = 0;
  for(int i=0; i<len; i++)
    val+=10;
  if(val != gpuData[0]){
    cerr<<"atomicAdd failed"<<endl;
    exit(EXIT_FAILURE);
  }else{
    cout<<"atomicAdd success"<<endl;
  }

  //sub
  val=0;
  for(int i=0; i<len; i++)
    val-=10;
  if(val != gpuData[1]){
    cerr<<"atomicSub failed"<<endl;
    exit(EXIT_FAILURE);
  }

 //exchange
 bool found = false;
 for(int i=0; i<len; i++)
   if(i == gpuData[2]){
     found=true;
     break;
   } 
 if(!found){
   cerr<<"atomicExch failed"<<endl;
   exit(EXIT_FAILURE);
 }

 //maximum
 val = -(1<<8);
 for(int i=0; i<len; i++)
    val = max(val, i);
 if(val != gpuData[3]){
   cerr<<"atomicMax failed"<<endl;
   exit(EXIT_FAILURE);
 }

 

 //minimum
 val = 1<<8;
 for(int i=0; i<len; i++)
   val = min(val,i);
 if(val != gpuData[4]){
    cerr<<"atomicMin failed"<<endl;
    exit(EXIT_FAILURE);
 } 

 // increment
 int limit=17;
 val=0;
 for(int i=0; i<len; i++)
    val = (val>=limit)? 0 : val+1;
 if(val != gpuData[5]){
    cerr<<"atomicInc failed"<<endl;
    exit(EXIT_FAILURE);
 }

 // decrement
 limit = 137;
 val = 0;
 for(int i=0; i<len; i++)
   val = ((val==0)||(val>limit)) ? limit : val-1;
 if(val != gpuData[6]){
    cerr<<"atomicDec failed"<<endl;
    exit(EXIT_FAILURE);
 } 

 //compare and swap
 found = false;
 for(int i=0; i<len ; i++)
   if(i == gpuData[7]){
     found = true;
     break;
   }
 if(!found){
    cerr<<"atomicCAS failed"<<endl;
    exit(EXIT_FAILURE);
 }
 
 // Bit
 // AND
 val = 0xFF;
 for(int i=0; i<len; i++)
   val &= (2*i+7);
 if(val != gpuData[8]){
    cerr<<"atomicAnd failed"<<endl;
    exit(EXIT_FAILURE);
 }
 

 // OR
 val = 0;
 for(int i=0; i<len; i++)
  val |= (1<<i);
 if(val != gpuData[9]){
   cerr<<"atomicOr failed"<<endl;
   exit(EXIT_FAILURE);
 }

 // XOR
 val  = 0xFF;
 for(int i=0; i<len ; i++){
   val ^= i;
 }
 if(val != gpuData[10]){
   cerr<<"atomicXor failed"<<endl;
   exit(EXIT_FAILURE);
 }
  
 return 0;  
}
