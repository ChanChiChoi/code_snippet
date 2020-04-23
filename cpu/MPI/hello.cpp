#include<iostream>
#include<string>
#include<cstring>
#include"mpi.h"

using namespace std;

int main(int argc, char* argv[]){

  int numprocs, myid, source;
  MPI_Status status;
  char mes[100];
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if(myid != 0){//非0 号进程发送消息
    strcpy(mes,"Hello World!");
    MPI_Send(mes, strlen(mes)+1, MPI_CHAR, 0, 99, MPI_COMM_WORLD);
  }else{//0 号进程接收消息
    for(source = 1; source<numprocs; source++){
      MPI_Recv(mes, 100, MPI_CHAR, source, 99, MPI_COMM_WORLD, &status);
      cout<<"来自进程:"<<source<<" 的消息:"<<mes<<endl;
    }
  }

  MPI_Finalize();
}
