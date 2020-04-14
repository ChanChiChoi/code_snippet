#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

/*
 * x+2y-z=3;
 * 2x+y-2z=3;
 * -3x+y+z=-6;
 *
 */
MatrixXd func(MatrixXd& mat){
  mat(0,0)=1;  mat(0,1)=2; mat(0,2)=-1; mat(0,3)=3;
  mat(1,0)=2;  mat(1,1)=1; mat(1,2)=-2; mat(1,3)=3;
  mat(2,0)=-3; mat(2,1)=1; mat(2,2)=1;  mat(2,3)=-6;
  return mat;
}

 
int
main(){
  
  MatrixXd mat(3,4);
  func(mat);
  cout<<mat<<endl;
}
