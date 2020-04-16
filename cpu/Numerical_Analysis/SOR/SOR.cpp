#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * x+2y-z=3;
 * 2x+y-2z=3;
 * -3x+y+z=-6;
 *
 */
void func(MatrixXd& mat, VectorXd& b){
  mat(0,0)=3;  mat(0,1)=1; mat(0,2)=-1; 
  mat(1,0)=2;  mat(1,1)=4; mat(1,2)=1; 
  mat(2,0)=-1; mat(2,1)=2; mat(2,2)=5;  

  b(0)=4; b(1)=1; b(2)=1;
}



void preprocess( MatrixXd& mat, MatrixXd& L, MatrixXd& D, MatrixXd& U){
  for(int row=0; row<mat.rows(); row++){
    for(int col=0; col<mat.cols(); col++){
      if(row<col){
        U(row,col) = mat(row,col);
      }else if(row==col){
        D(row,col) =  mat(row,col);
      }else{
        L(row,col) = mat(row,col);
      }
    }
  }
}


void SOR(VectorXd&x_i1, VectorXd& x_i,
            MatrixXd& L, MatrixXd& D, 
            MatrixXd& U, VectorXd& b,
            float omega){

  float epslion = 1e-9;
  size_t i=0;
  
  for(;;i++){
  // x_i1 =(1-omega)*x_i+ omega*D_inv*(b-U*x_i-L*x_i1);
  // 下面是SOR的公式
  // x_k1 = (wL+D)^(-1) [(1-w)Dx_k - wUx_k]+w(D+wL)^(-1)b
   x_i1 = (omega*L+D).inverse() * \
          ( (1-omega)*D*x_i - omega*U*x_i ) +\
          omega*(D+omega*L).inverse()*b;
   if(fabs((x_i1-x_i).norm())<epslion)
     break;

   x_i = x_i1;

  }
  cout<<"iteration:"<<i<<endl;
}

 
int
main(){
  
  MatrixXd mat(3,3);
  MatrixXd L(3,3);
  MatrixXd D(3,3);
  MatrixXd U(3,3);
  VectorXd b(3);
  func(mat, b);
  preprocess(mat,L,D,U);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---b:"<<endl<<b<<endl;
  cout<<"---L:"<<endl<<L<<endl;
  cout<<"---D:"<<endl<<D<<endl;
  cout<<"---U:"<<endl<<U<<endl;
  float omega = 1.25;
  cout<<"--omega:"<<endl<<omega<<endl;
  cout<<"---SOR---"<<endl;
  VectorXd x_i(3);
  VectorXd x_i1(3);
  SOR(x_i1,x_i,L,D, U,b, omega);
  cout<<"---x:"<<endl<<x_i1<<endl;
} 
