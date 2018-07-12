import torch
import torch.nn as nn
from BiDAF import BiDAF
from DataLoader import DataLoader
import torch.nn.functional as F
import evaluate as eva
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import traceback
from PredAnswer import pred_answer
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", help="Set batch size. (Default 60)")
args = parser.parse_args()
# In[2]:

BATCH_SIZE = 60

if args.batch:
    BATCH_SIZE = int(args.batch)

print("Batch size: ", BATCH_SIZE)

# In[3]:


#Load dataset and generate iterators
dataloader = DataLoader(batch_size=BATCH_SIZE,device = 0)
dev = dataloader.dev()
vocab_vec = dataloader.vocab_vec.cuda()
char_size = dataloader.char_size




model = BiDAF(vocab_vec, char_size, batch_size=BATCH_SIZE)
model.load_state_dict(torch.load("./Model/parameters_cp_39000"))
model.cuda()

print("CUDA is available: ",torch.cuda.is_available())
print("CUDA: ",next(model.parameters()).is_cuda)

def count_parameters_total(model):
    return sum(p.numel() for p in model.parameters())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_cuda(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.is_cuda)

param_total = count_parameters_total(model)
param_grad = count_parameters(model)
param_cuda = count_parameters_cuda(model)
print("Number of paramerters in model: {} \nNumber of trainable parameters in model: {}".format(param_total, param_grad))
print("Number of CUDA paramerters in model: {}".format(param_cuda))

answers_out = {}
em_avg = 0
f1_avg = 0
for batch in tqdm(dev,ascii=True):
    if batch.context.shape[0] != 50:
        continue
    #Batch data
    with torch.no_grad():
        #dev set
        with open("./data/squad/dev.context") as f:
            ori_context = list(f)
        with open("./data/squad/dev.answer") as f:
            ori_answer = list(f)
        with open("./data/squad/dev.id") as f:
            ori_id = list(f)
        data = batch
        context = data.context.cuda()
        query = data.query.cuda()
        label = data.label.cuda()
        context_c = data.context_c.cuda()
        query_c = data.query_c.cuda()
        index = data.index.cuda()
        label_start = label.narrow(1,0,1).squeeze()
        label_end = label.narrow(1,1,1).squeeze()

        #Feed forward
        pred_start, pred_end = model(context, query, context_c, query_c)
        #loss
        predictions, answers = pred_answer(pred_start,pred_end, index, ori_context, ori_answer, BATCH_SIZE)
        for i in range(len(predictions)):
            qid = ori_id[int(index[i])].strip()
            pred = predictions[i]
            answers_out[qid]=pred
        val_stat = eva.train_eval(answers,predictions)
        val_em, val_f1 = val_stat["exact_match"], val_stat["f1"]
        em_avg += val_em*BATCH_SIZE
        f1_avg += val_f1*BATCH_SIZE

length = len(answers_out)
path = "./data/eval/dev-pred.json"
if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
with open(path, "w") as f:
    json.dump(answers_out,f)

print("{} questions answered!".format(length))
print("EM: {} , F1: {}".format(em_avg/length,f1_avg/length))
