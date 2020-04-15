#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat){
  mat(0,0)=4;  mat(0,1)=-2; mat(0,2)=2; 
  mat(1,0)=-2;  mat(1,1)=2; mat(1,2)=-4; 
  mat(2,0)=2; mat(2,1)=-4; mat(2,2)=11;  
}

bool isPositiveDefinite(MatrixXd& mat){
  if(mat.transpose()!=mat){
    cerr<<"不是对称矩阵"<<endl;
    return false;
  }
  auto eigen = mat.eigenvalues();

  for(int i=0; i<eigen.rows(); i++)
    if (eigen(i).real()<=0){
      cerr<<"有非正特征值"<<endl;
      return false;
    }
    
  return true;

}

void Cholesky(MatrixXd& A, MatrixXd& R){

  for(int k=0; k<A.rows(); k++){
    // if A_kk <0, stop
    if(A(k,k)<0)
      break;

    //R_kk = sqrt(A_kk)
    R(k,k) = sqrt(A(k,k));
    //当处理对角线最后一个时，下面的逻辑就不需要处理了。
    if (k>= A.rows()-1) continue;

    //u^T = 1/R_kk * A_[k,k+1:n]
    float _tmp = 1/R(k,k);
    
    int width = A.cols()-k-1;
    auto ABlock =  A.block(k,k+1,1,width);

    auto u = _tmp * ABlock;

    //R_[k,k+1:n] = u^T;
    R.block(k,k+1,1,width) = u;

    //A_[k+1:n, k+1:n] = A_[k+1:n, k+1:n] - u*u^T;
    int height = A.rows()-k-1;
    A.block(k+1,k+1,height,width) -= u.transpose()*u; 
  }
}

 
int
main(){
  
  MatrixXd mat(3,3);
  MatrixXd R(3,3);
  func(mat);

  isPositiveDefinite(mat);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---Cholesky---"<<endl;
  Cholesky(mat,R);
  cout<<"---R:"<<endl<<R<<endl;
} 
