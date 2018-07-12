import torch
import torch.nn as nn
from functools import reduce

def pred_answer(start, end, index, ori_context, ori_answer, batch_size):
    ori_context = list(map(lambda x:x.split(), ori_context))
    pred = []
    answ = []
    for i in range(batch_size):
        length = len(ori_context[index[i]])
        start_prob = start[i,:]
        end_prob = end[i,:]
        maxprob = 0
        start_ptr = 0
        end_ptr = 0
        for j in range(len(start_prob)):
            end_max, end_ptr_t = torch.max(end_prob[j:],0)
            if float(start_prob[j])*float(end_max) > maxprob:
                start_ptr = j
                end_ptr = end_ptr_t+j
                maxprob = float(start_prob[j])*float(end_max)
        if start_ptr >= length or int(end_ptr) >= length:
            continue
        pred.append(reduce(lambda x,y:x+" "+y,ori_context[index[i]][start_ptr:int(end_ptr)+1]))
        answ.append(ori_answer[index[i]])
    return pred,answ
