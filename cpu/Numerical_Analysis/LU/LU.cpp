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
void func(MatrixXd& mat,MatrixXd& L, VectorXd& b){
  mat(0,0)=1;  mat(0,1)=2; mat(0,2)=-1; 
  mat(1,0)=2;  mat(1,1)=1; mat(1,2)=-2; 
  mat(2,0)=-3; mat(2,1)=1; mat(2,2)=1;  

  L(0,0)=1; L(1,1)=1; L(2,2)=1;

  b(0)=3; b(1)=3; b(2)=-6;

}

//朴素高斯消去法
void LU( MatrixXd& L, MatrixXd& U){
  //从第二行开始,
  for(int pivotRow = 1; pivotRow<U.rows(); pivotRow++){
    //主元列
    int pivot = pivotRow-1;
    //后续每一行都得操作一遍
    for(int row=pivotRow; row<U.rows(); row++){
      //主元倍数
      float scale = U(row,pivot)/U(pivotRow-1,pivot);
      L(row,pivot) = scale;    
      for(int col=pivot; col<U.cols(); col++){
          U(row,col) -= scale*U(pivot,col);    
      }
    }
  }
}

void Lc_b(MatrixXd& L, VectorXd& c, VectorXd& b){ 

  for(int row=0; row<L.rows(); row++){
    float sum = 0;
    int col = 0;
    for(; col <row; col++)
      sum += c(col)*L(row,col);

    c(row) = (b(row) - sum) / L(row,col) ;
  }

}


void Ux_c(MatrixXd& U, VectorXd& x, VectorXd& c){ 

  for(int row=U.rows()-1; row>=0; row--){
    float sum = 0;
    int col = U.cols()-1;
    for(; col >row; col--)
      sum += x(col)*U(row,col);

    x(row) = (c(row) - sum) / U(row,col) ;
  }

}

 
int
main(){
  
  MatrixXd mat(3,3);
  MatrixXd L(3,3);
  MatrixXd U(3,3);
  VectorXd b(3);
  func(mat, L, b);
  U = mat;
  cout<<mat<<endl;
  cout<<"---LU---"<<endl;
  LU(L,U);
  cout<<"---mat:"<<endl<<mat<<endl;
  cout<<"---L:"<<endl<<L<<endl;
  cout<<"---U:"<<endl<<U<<endl;
  cout<<"---Lc=b---"<<endl;
  VectorXd c(b.rows());
  Lc_b(L,c,b);
  cout<<"---c:"<<endl<<c<<endl;
  VectorXd x(b.rows());
  cout<<"---Ux=c---"<<endl;
  Ux_c(U,x,c);
  for(int i=0; i<x.rows(); i++)
    cout<<"x"<<i<<"="<<x(i)<<endl;
} 
