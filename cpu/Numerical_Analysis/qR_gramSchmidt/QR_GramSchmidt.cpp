#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat ){
    mat(0,0)=1;  mat(0,1)=-4; mat(0,2)=1; 
    mat(1,0)=2;  mat(1,1)=3;  mat(1,2)=0; 
    mat(2,0)=2;  mat(2,1)=2;  mat(2,2)=0;  
}

void GramSchmidt(MatrixXd& A, MatrixXd& Q, MatrixXd& R){
  // y_j = A_j - q1(q1^T*A_j) - q2(q2^T*A_j) - ...q_(j-1)*(q_(j-1)*A_j)
  // q_j = y_j / ||y_j||_2;
  for(int col=0; col<A.cols(); col++){

     VectorXd A_col = A.block(0,col, A.rows(),1); 
     VectorXd y = A_col;

     for(int i=0; i<col; i++){
       VectorXd _q = Q.block(0,i,Q.rows(),1);
       y -=  _q*(_q.transpose()*A_col);
     }
     // q_i = y_i / ||y_i||_2;
     auto q = y / y.norm();
     Q.block(0,col,Q.rows(),1) = q;

     //----------------------------
     for(int row=0;row<=col;row++){
        if(row==col)
          R(row,col) = y.norm();
        else{
          VectorXd _q = Q.block(0,row,Q.rows(),1);
          R(row,col) = _q.transpose()*A_col;
        }
     }
  }

}

 
int
main(){
 
  constexpr int m=3; 
  constexpr int n=2;
  // m > n 表示方程组的个数大于变量的个数，会造成无解
  // 完全的斯密特正交，是通过在A_j中加上m-n个额外的向量，因而m向量可以张成R^m
  if (m>n){
    cout<<"m>n,故而在A最右边增加向量(随便什么向量都行，反正会被正交)"<<endl; 
  }
  MatrixXd mat(m,m);
  func(mat);
  MatrixXd Q(m,m);
  MatrixXd R(m,m);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---GramSchmidt---"<<endl;
  GramSchmidt(mat,Q,R);
  cout<<"---Q:"<<endl<<Q<<endl;  
  cout<<"---R:"<<endl<<R<<endl;  
  cout<<"---backward---"<<endl;
  cout<<"---Q*R[:,:n] = A"<<endl;
  cout<<Q*R.block(0,0,R.rows(),n)<<endl;
} 
