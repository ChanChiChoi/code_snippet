#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

#include "Normal.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat, VectorXd& b){
  mat(0,0)=3;  mat(0,1)=1; mat(0,2)=-1; 
  mat(1,0)=2;  mat(1,1)=4; mat(1,2)=1; 
  mat(2,0)=-1; mat(2,1)=2; mat(2,2)=5;  

  b(0)=4; b(1)=1; b(2)=1;
}


void precondKrylov(MatrixXd& A, VectorXd& b, VectorXd& x, MatrixXd& M){

  int rA = A.rows();
  int cA = A.cols();
  int m = min(5,cA);

  VectorXd r = b - A*x;
  r = M*r; //乘以预条件

  // 存放m次正交计算
  MatrixXd q = MatrixXd::Zero(rA,m);
  q.block(0,0,rA,1) =  r / r.norm(); // q1 = r/ ||r||_2

  MatrixXd H = MatrixXd::Zero(m+1,m);// [k+1, k]
  VectorXd x0 = x;

  for(int k=0;k<m;k++){
    //y = A*q_k
    VectorXd y = A*q.block(0,k,rA,1);
    y = M*y;
    
    for(int j=0;j< k+1;j++){//必须可以访问到
      // q_j^T * y;
      auto _tmp = q.block(0,j,rA,1).transpose()*y;
      H(j,k) =  _tmp(0,0);
      // h_jk * q_j
      y -= H(j,k)*q.block(0,j,rA,1);
    }  
    
    H(k+1,k) = y.norm();
    
    if(fabs(H(k+1,k))>1e-6)// 如果不等于0, 存储新的列
      q.block(0,k+1,rA,1) = y / H(k+1,k);

    //用法线方程解决||Hc_k - [||r||_2,0,0...]^T ||_2
    VectorXd _b = VectorXd::Zero(k+2);
    _b(0) = r.norm();

    VectorXd c = VectorXd::Zero(k+1);
    
    MatrixXd subH = H.block(0,0,k+2,k+1); 
    Normal(subH,_b,c);
    //x_k = Q_k*c_k + c0;
    x = q.block(0,0,rA,k+1)*c + x0;
   
    cout<<"the ["<<k+1<<"] th solution:"<<endl<<x<<endl;
  }
}

 
int
main(){
  
  MatrixXd mat(3,3);
  VectorXd b(3);
  func(mat, b);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---b:"<<endl<<b<<endl;

  cout<<"---precondkrylov---"<<endl;

  VectorXd x1 = VectorXd::Zero(3);
  cout<<"---Jacobi precond:"<<endl;
  VectorXd diag = mat.diagonal();
  MatrixXd D = diag.asDiagonal();
  cout<<"---D:"<<endl<<D<<endl;
  MatrixXd M = D.inverse();//Jacobi 预条件
  cout<<"---M_inv:"<<endl<<M<<endl;
  precondKrylov(mat,b,x1,M);
  cout<<"true:2 -1 1"<<endl;
  cout<<"---x:"<<endl<<x1<<endl;


  VectorXd x2 = VectorXd::Zero(3);
  cout<<"---Gaussian Seridel precond:"<<endl;
  float omega = 1;
  MatrixXd L = mat.triangularView<Eigen::Lower>();
  L -= D;
  cout<<"---L:"<<endl<<L<<endl;
  MatrixXd U = mat.triangularView<Eigen::Upper>();
  U -= D;
  cout<<"---U:"<<endl<<U<<endl;
  M = (D+omega*L)*D.inverse()*(D+omega*U);
  cout<<"---M:"<<endl<<M<<endl;

  precondKrylov(mat,b,x2,M);
  cout<<"true:2 -1 1"<<endl;
  cout<<"---x:"<<endl<<x2<<endl;

} 
