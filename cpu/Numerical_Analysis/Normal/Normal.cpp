#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

#include "Normal.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat, VectorXd& b){
  mat(0,0)=1;  mat(0,1)=1;
  mat(1,0)=1;  mat(1,1)=-1;
  mat(2,0)=1; mat(2,1)=1; 

  b(0)=2; b(1)=1; b(2)=3;
}


 
int
main(){
  
  MatrixXd mat(3,2);
  VectorXd b(3);
  func(mat, b);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---b:"<<endl<<b<<endl;
  cout<<"---Normal---"<<endl;
  VectorXd x=VectorXd::Zero(2);
  Normal(mat,b,x);
  cout<<"---x:"<<endl<<x<<endl;
} 
