#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
  string image_path = "1.jpg";
  Mat img = imread(image_path, IMREAD_COLOR);  
  if(img.empty()){
    cout<<"Could not read the image:"<<image_path<<endl;
    return 1;
  }
  namedWindow("Display window", WINDOW_AUTOSIZE);
  imshow("Display window", img);
  int k = waitKey(0);
  if(k == 's'){
    imwrite("1.png",img);
  }
  return 0;
}
