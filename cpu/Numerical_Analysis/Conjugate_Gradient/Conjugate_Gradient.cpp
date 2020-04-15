#include<iostream>
#include<iomanip>
#include<cmath>
#include<Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void func(MatrixXd& mat, VectorXd& b, VectorXd& x){
  mat(0,0)=2;  mat(0,1)=2;// mat(0,2)=2; 
  mat(1,0)=2;  mat(1,1)=5;// mat(1,2)=-4; 
//  mat(2,0)=2; mat(2,1)=-4; mat(2,2)=11;  

  b(0)=6; b(1)=3;
  x(0)=0; x(1)=0;
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

void Conjugate_Gradient(MatrixXd& A, VectorXd& b, VectorXd& x){

  //Ax_k1 + r_k1 = A(x_k + alpha_k*d_k ) + r_k - alpha_k*A*d_k
  //             = Ax_k + r_k;
  // 其中 x_k1 = x_k + alpha_k*d_k;
  // 该方法能够成功就在于所有余项都和前面的余项正交，如果能做到所有的余项正交，
  // 该方法搜索所有的正交方向，在经过至多n步就可以得到余项为零的正确解
  // 所以余项的正交关键在于选择搜索方向d_k，使之两两共轭
  VectorXd r_k = b - A*x; // 表示近似解的余项
  VectorXd d = r_k; //表示用于更新x_k 得到改进的x_k1时使用的新的搜索方向
  VectorXd r_k1 = r_k;
  auto zero = VectorXd::Zero(r_k.rows()); 

  for(int k=0; k<b.rows(); k++){
   if(r_k == zero) break;

    double _alpha0 = r_k.transpose()*r_k ;
    double _alpha1 = d.transpose()*A*d;
    double alpha = _alpha0 / _alpha1;

    x += alpha*d; // 更新x
    r_k1 =  r_k - alpha*A*d;// 获得r_k1
    
    double _beta0 = r_k1.transpose()*r_k1;
    double _beta1 = r_k.transpose()*r_k;
    double beta = _beta0 / _beta1;

    d = r_k1 + beta*d; // 更新d

    r_k = r_k1; // 更新r_k 

  }
}

 
int
main(){
  
  MatrixXd mat(2,2);
  VectorXd b(2);
  VectorXd x(2);
  func(mat,b,x);
  cout<<b.rows()<<endl;
  isPositiveDefinite(mat);
  VectorXd x0 = x;
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---Conjugate_Gradient---"<<endl;
  Conjugate_Gradient(mat,b,x);
  cout<<"---x:"<<endl<<x<<endl;
} 
