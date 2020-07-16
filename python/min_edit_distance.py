'''
动态规划的路子实现最小编辑距离
'''
import numpy as np

def del_cost(char):
  return 1

def ins_cost(char):
  return 1

def sub_cost(src,dst):
  if src == dst: return 0
  else: return 2
  



def min_edit_distance(src:str,dst:str) -> np.ndarray:
  n = len(src)
  m = len(dst)
  
  D = np.zeros([n+1,m+1])

  # initialization
  for i in range(1,n+1):
    D[i,0] = D[i-1,0] + del_cost(src[i-1])

  for j in range(1,m+1):
    D[0,j] = D[0,j-1] + ins_cost(dst[j-1])

  # recurrence relation:
  for i in range(1,n+1):
    for j in range(1,m+1):
      D[i,j] = min(D[i-1,j]+del_cost(src[i-1]),
                   D[i-1,j-1]+sub_cost(src[i-1],dst[j-1]),
                   D[i,j-1]+ins_cost(dst[j-1]))

  return D


if __name__ == '__main__':
  src = 'intention'
  dst = 'execution'
  distance = min_edit_distance(src,dst)
  print(distance)
  print('D[n,m]:',distance[-1,-1])
