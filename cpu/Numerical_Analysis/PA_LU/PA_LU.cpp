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
void func(MatrixXd& mat,MatrixXd& P, VectorXd& b){
  mat(0,0)=2;  mat(0,1)=1; mat(0,2)=5; 
  mat(1,0)=4;  mat(1,1)=4; mat(1,2)=-4; 
  mat(2,0)=1; mat(2,1)=3; mat(2,2)=1;  

  P(0,0)=1; P(1,1)=1; P(2,2)=1;

  b(0)=5; b(1)=0; b(2)=6;

}


int getMaxRow(MatrixXd& mat, int pivotCol){
  int ind = pivotCol;
  float maxVal = mat(pivotCol, pivotCol);
  for(int row=pivotCol+1; row<mat.rows(); row++){
    if (mat(row,pivotCol)>maxVal){
      ind = row;
      maxVal = mat(row,pivotCol);
    }
  }
  return ind;
}

void change(int pivotRow,int maxRow, MatrixXd& mat){
  if (pivotRow == maxRow) return;
  for(int i=0;i<mat.cols(); i++){
    //swap P
    float tmp = mat(pivotRow,i);
    mat(pivotRow,i) = mat(maxRow,i);
    mat(maxRow,i) = tmp;
  }
}

void PA_LU( MatrixXd& P, MatrixXd& L, MatrixXd& U){
  //从第二行开始,
  for(int pivotRow = 1; pivotRow<U.rows(); pivotRow++){
    //主元列
    int pivotCol = pivotRow-1;
    //增加换行操作
    int maxRow = getMaxRow(U, pivotCol);
    change(pivotRow-1,maxRow,P);
    change(pivotRow-1,maxRow,U);
    change(pivotRow-1,maxRow,L);
    //后续每一行都得操作一遍
    for(int row=pivotRow; row<U.rows(); row++){
      //主元倍数
      float scale = U(row,pivotCol)/U(pivotRow-1,pivotCol);
      L(row,pivotCol) = scale;    
      for(int col=pivotCol; col<U.cols(); col++){
          U(row,col) -= scale*U(pivotCol,col);    
      }
    }
  }

  //填充L对角线
  for(int i=0; i<L.rows(); i++)
   L(i,i) = 1;
}

void Lc_Pb(MatrixXd& L, VectorXd& c, VectorXd& b){ 

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
  MatrixXd P(3,3);
  MatrixXd L(3,3);
  MatrixXd U(3,3);
  VectorXd b(3);
  func(mat, P, b);
  U = mat;
  cout<<"---A:"<<endl<<mat<<endl;
  cout<<"---P:"<<endl<<P<<endl;
  cout<<"---PA=LU---"<<endl;
  PA_LU(P,L,U);
  
  cout<<"---P:"<<endl<<P<<endl;
  cout<<"---L:"<<endl<<L<<endl;
  cout<<"---U:"<<endl<<U<<endl;
  cout<<"---Lc=Pb---"<<endl;
  VectorXd c(b.rows());
  b = P*b;
  Lc_Pb(L,c,b);
  cout<<"---c:"<<endl<<c<<endl;
  VectorXd x(b.rows());
  cout<<"---Ux=c---"<<endl;
  Ux_c(U,x,c);
  for(int i=0; i<x.rows(); i++)
    cout<<"x"<<i<<"="<<x(i)<<endl;
} 
