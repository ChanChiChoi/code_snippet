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
MatrixXd func(MatrixXd& mat){
  mat(0,0)=1;  mat(0,1)=2; mat(0,2)=-1; mat(0,3)=3;
  mat(1,0)=2;  mat(1,1)=1; mat(1,2)=-2; mat(1,3)=3;
  mat(2,0)=-3; mat(2,1)=1; mat(2,2)=1;  mat(2,3)=-6;
  return mat;
}

//朴素高斯消去法
MatrixXd gaussian(MatrixXd& mat){
  //从第二行开始,
  for(int pivotCol = 1; pivotCol<mat.rows(); pivotCol++){
    //主元列
    int pivot = pivotCol-1;
    //后续每一行都得操作一遍
    for(int row=pivotCol; row<mat.rows(); row++){
      //主元倍数
      float scale = mat(row,pivot)/mat(pivotCol-1,pivot);
      
      for(int col=pivot; col<mat.cols(); col++){

          mat(row,col) -= scale*mat(pivot,col);    
      }
    }
  }
  return mat;
}

VectorXd backward(MatrixXd& mat, VectorXd& x){ 

  for(int row=mat.rows()-1; row>=0; row--){
    // col 减2 是因为cols比rows大1
    float sum = 0;
    int col = mat.cols()-2;
    for(; col >row; col--)
      sum += x(col)*mat(row,col);

    x(row) = (mat(row,mat.cols()-1) - sum) / mat(row,col) ;
  }
  return x;

}


 
int
main(){
  
  MatrixXd mat(3,4);
  func(mat);
  cout<<mat<<endl;
  cout<<"---gaussian---"<<endl;
  gaussian(mat);
  cout<<mat<<endl;
  cout<<"---solution---"<<endl;
  VectorXd x(3);
  backward(mat,x);  
  for(int i = 0; i<x.rows();i++)
   cout<<"x"<<i<<"="<<x(i)<<endl;
} 
