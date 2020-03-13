#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;



int Partition(vector<int> &vec, int left,int right){
  // get current value
  // left part is not bigger than cur,
  //right part is not smaller than cur.
  int tmp = vec[left];
  while(left<right){
    // skip the value not smaller than current from right side
    while(left<right && vec[right]>=tmp)
      right--;
    //move small or equal value to left side
    vec[left]=vec[right];

    //skip the value not bigger than current from left side
    while(left<right && vec[left]<=tmp)
     left++;
    // move bigger or equal value to right side
    vec[right]=vec[left];
  }
  //current part should be assigned
  vec[left]=tmp;
  // now, left side of 'vec[left]' is smaller than or equal of tmp
  // right side of 'vec[left]' is bigger than or equal of tmp 
  return left;
}


void findk(vector<int>& input, int k, int left, int right){

  int ind;
  ind = Partition(input, left,right);
  if(ind == k-1)
    cout<<"the "<<ind+1<<" th num is:"<<input[ind]<<endl;
  else if(ind>k-1)
    findk(input,k,left,ind-1);
  else
    findk(input,k,ind+1,right);
}


int main(){
  vector<int> input{15,25,9,48,36,100,58,99,126,5};

  //1-input data
  cout<<"input data:";
  for(auto &x:input)
    cout<<x<<" ";
  cout<<endl;

  //2-input kth
  int k;
  cout<<"input kth:";
  cin>>k;
  cout<<endl;

  //3-find kth
  findk(input,k,0, input.size()-1);
  
  //4-sort the vector, just for display  
  sort(input.begin(), input.end());
  for(auto &x:input)
    cout<<x<<" ";
  cout<<endl;
  return 0;
}
