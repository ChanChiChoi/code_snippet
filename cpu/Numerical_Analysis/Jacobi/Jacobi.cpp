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
void func(MatrixXd& mat, VectorXd& b){
  mat(0,0)=3;  mat(0,1)=1; mat(0,2)=-1; 
  mat(1,0)=2;  mat(1,1)=4; mat(1,2)=1; 
  mat(2,0)=-1; mat(2,1)=2; mat(2,2)=5;  

  b(0)=4; b(1)=1; b(2)=1;
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

void preprocess( MatrixXd& mat, MatrixXd& L, MatrixXd& D_inv, MatrixXd& U){
  for(int row=0; row<mat.rows(); row++){
    for(int col=0; col<mat.cols(); col++){
      if(row<col){
        U(row,col) = mat(row,col);
      }else if(row==col){
        D_inv(row,col) = 1 / mat(row,col);
      }else{
        L(row,col) = mat(row,col);
      }
    }
  }
}


void Jacobi(VectorXd&x_i1, VectorXd& x_i,
            MatrixXd& L, MatrixXd& D_inv, 
            MatrixXd& U, VectorXd& b){

  float epslion = 1e-9;
  size_t i=0;

  for(;;i++){

   x_i1 = D_inv*(b-U*x_i-L*x_i);

   if(fabs((x_i1-x_i).norm())<epslion)
     break;

   x_i = x_i1;

  }
  //cout<<"iteration:"<<i<<endl;
}

 
int
main(){
  
  MatrixXd mat(3,3);
  MatrixXd L(3,3);
  MatrixXd D_inv(3,3);
  MatrixXd U(3,3);
  VectorXd b(3);
  func(mat, b);
  preprocess(mat,L,D_inv,U);
  cout<<"---mat:"<<endl<<mat<<endl;  
  cout<<"---L:"<<endl<<L<<endl;
  cout<<"---D_inv:"<<endl<<D_inv<<endl;
  cout<<"---U:"<<endl<<U<<endl;
  cout<<"---Jacobi---"<<endl;
  VectorXd x_i(3);
  VectorXd x_i1(3);
  Jacobi(x_i1,x_i,L,D_inv, U,b);
  cout<<"---x:"<<endl<<x_i1<<endl;
} 
