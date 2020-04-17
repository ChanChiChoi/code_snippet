#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

#include "Normal_LM.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat, VectorXd& X){
  mat(0,0)=1;  mat(0,1)=3; 
  mat(1,0)=2;  mat(1,1)=5; 
  mat(2,0)=2;  mat(2,1)=7; 
  mat(3,0)=3;  mat(3,1)=5; 
  mat(4,0)=4;  mat(4,1)=1; 

  X(0)=1; X(1)=1; X(2)=1;
}


void getAfromJacobi(MatrixXd& A, MatrixXd& points, VectorXd& X){

  for(int row=0; row<A.rows(); row++){
    A(row,0) = exp(-X(1)*(points(row,0)-X(2))*(points(row,0)-X(2)));
    A(row,1) = -X(0)*(points(row,0)-X(2))*(points(row,0)-X(2))*exp(-X(1)*(points(row,0)-X(2))*(points(row,0)-X(2)));
    A(row,2) = 2*X(0)*X(1)*(points(row,0)-X(2))*exp(-X(1)*(points(row,0)-X(2))*(points(row,0)-X(2)));
  }
}

float f(float c1, float c2, float c3, float t, float y){
  
  return c1*exp(-c2*(t-c3)*(t-c3)) - y;
}

void get_r(VectorXd& r, MatrixXd& points,VectorXd& X){
    r(0) = f(X(0),X(1),X(2),points(0,0), points(0,1));
    r(1) = f(X(0),X(1),X(2),points(1,0), points(1,1));
    r(2) = f(X(0),X(1),X(2),points(2,0), points(2,1));
    r(3) = f(X(0),X(1),X(2),points(3,0), points(3,1));
    r(4) = f(X(0),X(1),X(2),points(4,0), points(4,1));

}

//-----------------------------------
void Levenberg_Marquardt(MatrixXd& A, MatrixXd& points, VectorXd& X){

  float epsilon = 1e-6;
  float lambda = 50;
  VectorXd xpre = X;
  int i=0;
  for(;;i++){
    // 用方程组，计算Jacobi矩阵
    getAfromJacobi(A,points,X);
    cout<<"["<<i<<"]th A:"<<endl<<A<<endl;
    //将当前的迭代值放入方程组，获取每个方程的值
    VectorXd r(points.rows());
    get_r(r,points,X) ;
    //用法线方程计算下面式子
    // A_T*A*v^k = -A^T * r(x^k)
    // 然后这就是法线方程计算的过程，所以不需要事先左右两边乘以A^T
    r *= -1;
    cout<<"["<<i<<"]th r:"<<endl<<r<<endl;

    VectorXd v = VectorXd::Zero(X.rows());
    Normal_LM(A,lambda,r,v); 
    cout<<"["<<i<<"]th v:"<<endl<<v<<endl;
    //进行迭代 x^(k1) = x^k + v^k;
    X += v;
    if ((X-xpre).norm()<epsilon )
        break;
    
    xpre = X;
  }
  
}

 
int
main(){
  
  MatrixXd points(5,2);
  VectorXd X(3);
  func(points,X);
  cout<<"---mat:"<<endl<<points<<endl;  
  cout<<"---X:"<<endl<<X<<endl;  
  cout<<"---Gaussian Newton---"<<endl;
  MatrixXd A = MatrixXd::Zero(5,3);
  Levenberg_Marquardt(A,points,X);
  cout<<"---x:"<<endl<<X<<endl;
} 
