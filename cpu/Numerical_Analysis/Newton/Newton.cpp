#include<iostream>
#include<iomanip>
#include<cmath>

using namespace std;

float const epsilon = 1e-9;

//原始函数：x^3+x+1 = 0;
float func(float x){
  return x*x*x+x-1;
}

//原始函数的导数
float func_der(float x){
  return 3*x*x+1;
}

//牛顿迭代公式
float newton(float x_i){
  float x_i1;
  x_i1 = x_i - func(x_i)/func_der(x_i);
  return x_i1;
}


int main(){

  float x_i = -0.7;
  float e=0;
  for(int i=0;;i++){

    float x_i1 = newton(x_i);
    e = fabs(x_i1 - x_i);
    cout<<"迭代["<<i<<"]次: x_i="
        <<setw(8)<<x_i<<"     x_i1="<<x_i1
        <<" 连续两次误差："<<e<<endl;
    if (e < epsilon)
      break;
    x_i = x_i1;
  }

}
