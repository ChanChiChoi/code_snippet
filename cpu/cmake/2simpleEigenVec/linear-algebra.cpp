#include<chrono>
#include<iostream>
#include<Eigen/Dense>

using Eigen::VectorXd;
using namespace std::chrono;
//using system_clock;
using std::cout;
using std::endl;

EIGEN_DONT_INLINE
double simple_function(Eigen::VectorXd &va, Eigen::VectorXd &vb){
  double d = va.dot(vb);
  return d;
}

int
main(){
  int len = 1000000;
  int num_repetitions = 100;
  
  VectorXd va = VectorXd::Random(len);
  VectorXd vb = VectorXd::Random(len);

  double result;
  auto st = system_clock::now();
  for(auto i=0; i<num_repetitions; i++)
    result = simple_function(va,vb);

  auto ed = system_clock::now();
  auto elapsed_seconds = ed - st;

  cout<<"result: "<<result<<endl;
  cout<<"elapsed seconds: "<<elapsed_seconds.count()<<endl;
}
