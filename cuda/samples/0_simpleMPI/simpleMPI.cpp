#include<iostream>
#include<iomanip>
#include<mpi.h>


#include"simpleMPI.h"

using namespace std;

#define MPI_CHECK(call) \
        if((call) != MPI_SUCCESS){ \
           cerr<<"MPI error calling [" #call "]\n"; \
            my_abort(-1); }


void my_abort(int err){
  cout<<"Result: FAILED"<<endl;
  MPI_Abort(MPI_COMM_WORLD, err);
}


int
main(int argc, char *argv[]){

  int blockSize = 256;
  int gridSize = 10000;
  int dataSizePerNode = gridSize*blockSize;
  //mpi初始化
  MPI_CHECK(MPI_Init(&argc, &argv));

  //获取命令行传递的节点数目和节点索引
  int commSize, commRank;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

  //在根节点上准备好所有数据(node 0)
  int dataSizeTotal = dataSizePerNode*commSize;
  float *dataRoot = nullptr;

  if(commRank == 0){// node 0 才需要数据准备
    cout<<"一共 "<<commSize<<" 节点"<<endl;
    dataRoot = new float[dataSizeTotal];
    initData(dataRoot, dataSizeTotal);
  }

  //每个节点分配一个buffer,包含0节点
  float *dataNode = new float[dataSizePerNode];

  //dispatch 输入数据的某个部分到每个节点
  MPI_CHECK(MPI_Scatter(dataRoot,
                        dataSizePerNode,
                        MPI_FLOAT,
                        dataNode,
                        dataSizePerNode,
                        MPI_FLOAT,
                        0,
                        MPI_COMM_WORLD));

  if(commRank == 0){ // root 节点不需要数据
    delete [] dataRoot;
  }

  //每个节点，各自在GPU上运行
  computeGPU(dataNode, blockSize, gridSize);

  //归约到root 节点，计算每个节点的输出值的和
  float sumNode = sum(dataNode, dataSizePerNode);
  cout<<"当前节点索引:"<<commRank
      <<" ;结果值为:"<<setprecision(10)
      <<setw(10)<<sumNode<<endl;
  float sumRoot;

  MPI_CHECK(MPI_Reduce(&sumNode, 
                       &sumRoot,
                       1,
                       MPI_FLOAT,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD));

  if(commRank == 0){
    float average = sumRoot / dataSizeTotal;
    cout<<"Average of square roots is: "<<average<<endl;
  }

  //clean
  delete [] dataNode;
  MPI_CHECK(MPI_Finalize());
  
  if(commRank == 0){
    cout<<"Result: PASSED"<<endl;
  }

  return 0;
}
