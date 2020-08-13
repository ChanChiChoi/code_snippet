# -*- coding: utf-8 -*-
# speech and language process 3rd [Figure 10.5,Figure 10.6]

import secrets

def new_state(vocabulary,state,word,y):
    return secrets.choice(vocabulary)


def beam_decode(c,beam_width=4):

    # 1 - 候选词集合
    vocabulary = ['a','b','c','d','e','f','EOF'] # vocabulary    

    # 2 - 第一次用encoder的上下文预测的softmax，然后选取排序前beam_width个词
    frontier = vocabulary[:beam_width]

    while frontier and beam_width:

      extended_froniter = []

      # frontier中每个词会与整个词组进行比对
      for state in frontier:
        # 3 - 每个词再进一次解码器，得到一个sofmax，y就是
        # y = decode(state)
        for word in vocabulary:
          successor = new_state(vocabulary,state,word,y)
        # 4 - 将len(frontier)*len(vocabulary)个组合进行排序，选取前beam_width个结果
        
        # 5 - 如果遇到EOF，则beam_width -= 1




    return best_paths 
