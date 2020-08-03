# -*- coding: utf-8 -*-
import numpy as np

'''
HMM模型，其中隐藏状态为词性，输出为字,即

NNP  -->   MD    -->    VB    -> DT   -> NN
|          |            |        |       |
Janet      will         back     the     bill

'''
'''
transition_mat = 
    NNP    MD     VB     JJ     NN     RB     DT
<s> 0.2767 0.0006 0.0031 0.0453 0.0449 0.0510 0.2026
NNP 0.3777 0.0110 0.0009 0.0084 0.0584 0.0090 0.0025
MD  0.0008 0.0002 0.7968 0.0005 0.0008 0.1698 0.0041
VB  0.0322 0.0005 0.0050 0.0837 0.0615 0.0514 0.2231
JJ  0.0366 0.0004 0.0001 0.0733 0.4509 0.0036 0.0036
NN  0.0096 0.0176 0.0014 0.0086 0.1216 0.0177 0.0068
RB  0.0068 0.0102 0.1011 0.1012 0.0120 0.0728 0.0479
DT  0.1147 0.0021 0.0002 0.2157 0.4744 0.0102 0.0017
'''

'''
emission_mat = 
    Janet    will     back     the      bill
NNP 0.000032 0        0        0.000048 0
MD  0        0.308431 0        0        0
VB  0        0.000028 0.000672 0        0.000028
JJ  0        0        0.000340 0        0
NN  0        0.000200 0.000223 0        0.002337
RB  0        0        0.010446 0        0
DT  0        0        0        0.506099 0
'''

'''
上面2个矩阵为转移矩阵和输出矩阵
'''
transition_mat = np.array(
                         [[0.2767,0.0006,0.0031,0.0453,0.0449,0.0510,0.2026],
                          [0.3777,0.0110,0.0009,0.0084,0.0584,0.0090,0.0025],
                          [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041],
                          [0.0322,0.0005,0.0050,0.0837,0.0615,0.0514,0.2231],
                          [0.0366,0.0004,0.0001,0.0733,0.4509,0.0036,0.0036],
                          [0.0096,0.0176,0.0014,0.0086,0.1216,0.0177,0.0068],
                          [0.0068,0.0102,0.1011,0.1012,0.0120,0.0728,0.0479],
                          [0.1147,0.0021,0.0002,0.2157,0.4744,0.0102,0.0017]])

emission_mat = np.array(
                       [[0.000032,0,0,0.000048,0],
                        [0,0.308431,0,0,0],
                        [0,0.000028,0.000672,0,0.000028],
                        [0,0,0.000340,0,0],
                        [0,0.000200,0.000223,0,0.002337],
                        [0,0,0.010446,0,0],
                        [0,0,0,0.506099,0]])

def viterbi():
  state = ['NNP','MD','VB','JJ','NN','RB','DT']
  time_frame = 5
  ans_viterbi = np.zeros([len(state),time_frame])
  backpointer = np.zeros([len(state),time_frame])

  # 从<s>开始第一步
  for s in range(len(state)):
     ans_viterbi[s,0] = transition_mat[0,s]*emission_mat[s,0]
     backpointer[s,0] = 0
  print('curmaxID:【{}】'.format(ans_viterbi[:,0].argmax()+1))

  bestpath = [state[ans_viterbi[:,0].argmax()]]
  for t in range(1,time_frame):
    for s in range(len(state)):
      pre_vec = ans_viterbi[:,t-1]*transition_mat[1:,s]*emission_mat[s,t]
      pre_max,pre_argmax = pre_vec.max(),pre_vec.argmax()

      ans_viterbi[s,t] = pre_max
      backpointer[s,t] = pre_argmax
    bestpath.append(state[ans_viterbi[:,t].argmax()])
    print('curmaxID:【{}】'.format(ans_viterbi[:,t].argmax()+1))

  bestpathprob = ans_viterbi[:,-1].max()
  bestpathpointer = ans_viterbi[:,-1].argmax()

  return bestpath,bestpathprob
  

if __name__ == '__main__':
    print('NNP MD VB DT NN')
    bestpath,prob = viterbi()
    print("bestPath:",bestpath)
    print('prob:',prob)
