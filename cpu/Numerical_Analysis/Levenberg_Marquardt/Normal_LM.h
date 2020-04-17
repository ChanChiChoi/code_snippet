#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd gaussian(MatrixXd&);
VectorXd backward(MatrixXd& mat, VectorXd& x); 

//法线方程
void Normal_LM(MatrixXd& A,float lambda, VectorXd& b, VectorXd& x){
  MatrixXd AT_Ab = MatrixXd::Zero(A.rows(),A.cols()+1);
  AT_Ab.block(0,0,A.rows(), A.cols()) = A;
  AT_Ab.block(0,A.cols(),A.rows(),1) = b;
  AT_Ab = A.transpose()*AT_Ab;
  
  //LM新增部分
  MatrixXd ATA = A.transpose()*A;
  VectorXd diagVec = ATA.diagonal();
  auto diagMat = diagVec.asDiagonal();
  AT_Ab.block(0,0,diagMat.rows(), diagMat.cols()) += lambda*diagMat;


  gaussian(AT_Ab);
  backward(AT_Ab,x);
 
}


//朴素高斯消去法
MatrixXd gaussian(MatrixXd& mat){
  //从第二行开始,
  for(int pivotRow = 1; pivotRow<mat.rows(); pivotRow++){
    //主元列
    int pivot = pivotRow-1;
    //后续每一行都得操作一遍
    for(int row=pivotRow; row<mat.rows(); row++){
      //主元倍数
      float scale = mat(row,pivot)/mat(pivotRow-1,pivot);
      
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


