#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat ){
    mat(0,0)=1;  mat(0,1)=-4; //mat(0,2)=1; 
    mat(1,0)=2;  mat(1,1)=3;  //mat(1,2)=0; 
    mat(2,0)=2;  mat(2,1)=2;  //mat(2,2)=0;  
}

void HouseHolder(MatrixXd& A, MatrixXd& Q, MatrixXd& R){

  //因为A是[m,n], 所以Q必须是[m,m],R必须是[m,n]
  R = A;
  int rA = A.rows();
  int cA = A.cols();
 
  Q = MatrixXd::Identity(Q.rows(), Q.cols());
  //每次假设扔掉第一行，第一列,然后处理该子矩阵
  for(int col=0; col<cA; col++){
    //计算当前要处理的维度，就是不断的缩小子矩阵
    int dim = rA-col;
    // 读取当前列
    VectorXd x = R.block(col,col,dim,1);
    // 获得omega向量
    VectorXd omega = VectorXd::Zero(dim);
    omega(0) = x.norm();
    //计算得到u向量
    VectorXd u = omega - x;
    //计算P矩阵
    MatrixXd P = (u*u.transpose()) / (u.transpose()*u);
    // 当前H矩阵就是一个单元矩阵，当前子矩阵区域减去：I-2P
    MatrixXd H = MatrixXd::Identity(rA,rA);
    H.block(col,col,dim,dim) -= 2*P;
    // 因为H1H2...A=R,所以不断记录R的生成过程
    R = H*R;
    //同理，H1H2...H_(cA) 就是Q了
    Q = Q*H;
  }

}

 
int
main(){
 
  constexpr int m=3; 
  constexpr int n=2;
  // m > n 表示方程组的个数大于变量的个数，会造成无解
  // 完全的斯密特正交，是通过在A_j中加上m-n个额外的向量，因而m向量可以张成R^m
  if (m>n){
    cout<<"m>n"<<endl; 
  }
  MatrixXd mat(m,n);
  func(mat);
  MatrixXd Q(m,m);
  MatrixXd R(m,n);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---HouseHolder---"<<endl;
  HouseHolder(mat,Q,R);
  cout<<"---Q:"<<endl<<Q<<endl;  
  cout<<"---R:"<<endl<<R<<endl;  
  cout<<"---backward---"<<endl;
  cout<<"---Q*R = A"<<endl;
  cout<<Q*R<<endl;
} 
