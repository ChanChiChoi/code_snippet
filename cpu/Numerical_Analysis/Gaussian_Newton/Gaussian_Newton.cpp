#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

#include "Normal.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat, VectorXd& X,VectorXd& R){
  mat(0,0)=-1;  mat(0,1)=0; 
  mat(1,0)=1;  mat(1,1)=0.5; 
  mat(2,0)=1; mat(2,1)=-0.5; 

  X(0)=0; X(1)=0;
  R(0)=1; R(1)=0.5; R(2)=0.5;
}

float S(float x, float y, float xi, float yi){

  return sqrt((x-xi)*(x-xi)+
              (y-yi)*(y-yi));
}

void getAfromJacobi(MatrixXd& A, MatrixXd& points, VectorXd& X){
  A(0,0) =  (X(0) - points(0,0)) / S(X(0),X(1),points(0,0),points(0,1));
  A(0,1) =  (X(1) - points(0,1)) / S(X(0),X(1),points(0,0),points(0,1));

  A(1,0) =  (X(0) - points(1,0)) / S(X(0),X(1),points(1,0),points(1,1));
  A(1,1) =  (X(1) - points(1,1)) / S(X(0),X(1),points(1,0),points(1,1));

  A(2,0) =  (X(0) - points(2,0)) / S(X(0),X(1),points(2,0),points(2,1));
  A(2,1) =  (X(1) - points(2,1)) / S(X(0),X(1),points(2,0),points(2,1));
}

float d(float x, float y, float xi, float yi, float r){
  
  return S(x,y,xi,yi) - r;
}

void get_r(VectorXd& r, MatrixXd& points,VectorXd& X, VectorXd& R){
    r(0) = d(X(0),X(1),points(0,0), points(0,1),R(0));
    r(1) = d(X(0),X(1),points(1,0), points(1,1),R(1));
    r(2) = d(X(0),X(1),points(2,0), points(2,1),R(2));

}

//-----------------------------------
void Gaussian_Newton(MatrixXd& A, MatrixXd& points, VectorXd& X, VectorXd& R){

  float epsilon = 1e-6;
  VectorXd xpre = X;
  int i=0;
  for(;;i++){
    // 用方程组，计算Jacobi矩阵
    getAfromJacobi(A,points,X);
    cout<<"["<<i<<"]th A:"<<endl<<A<<endl;
    //将当前的迭代值放入方程组，获取每个方程的值
    VectorXd r(3);
    get_r(r,points,X,R) ;
    //用法线方程计算下面式子
    // A_T*A*v^k = -A^T * r(x^k)
    // 然后这就是法线方程计算的过程，所以不需要事先左右两边乘以A^T
    r *= -1;
    cout<<"["<<i<<"]th r:"<<endl<<r<<endl;

    VectorXd v = VectorXd::Zero(2);
    Normal(A,r,v); 
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
  
  MatrixXd points(3,2);
  VectorXd R(3);
  VectorXd X(2);
  func(points,X, R);
  cout<<"---mat:"<<endl<<points<<endl;  
  cout<<"---R:"<<endl<<R<<endl;  
  cout<<"---X:"<<endl<<X<<endl;  
  cout<<"---Gaussian Newton---"<<endl;
  MatrixXd A = MatrixXd::Zero(3,2);
  Gaussian_Newton(A,points,X,R);
  cout<<"---x:"<<endl<<X<<endl;
} 
