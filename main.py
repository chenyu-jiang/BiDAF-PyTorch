
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from layers/BiDAF import BiDAF
from DataLoader import DataLoader
import torch.nn.functional as F
import evaluate/evaluate as eva
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import traceback
from PredAnswer import pred_answer

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning_rate", help="Set learning rate. (Default 0.01)")
parser.add_argument("-b", "--batch", help="Set batch size. (Default 60)")
parser.add_argument("-s", "--step", help="Set step size. (Default 20000)")
args = parser.parse_args()
# In[2]:

LR = 0.001
BATCH_SIZE = 60
STEP = 20000

if args.learning_rate:
    LR = float(args.learning_rate)
if args.batch:
    BATCH_SIZE = int(args.batch)
if args.step:
    STEP = int(args.step)

print("Batch size: ", BATCH_SIZE)
print("Step: ", STEP)
print("Learning rate: ", LR)

# In[3]:


#Load dataset and generate iterators
dataloader = DataLoader(batch_size=BATCH_SIZE,device = 0)
vocab_vec = dataloader.vocab_vec.cuda()
char_size = dataloader.char_size


# In[4]:


#Initiate summary writer
writer = SummaryWriter("./log")


# In[5]:


model = BiDAF(vocab_vec, char_size, batch_size=BATCH_SIZE)
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


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr = LR, weight_decay=1e-4)
criterion = F.cross_entropy
# def bidaf_loss(pred_start, pred_end, label):
#     loss = torch.zeros(1)
#     for i in range(label.shape[0]):
#         label_start = label[i][0] if label[i][0]<400 else 399
#         label_end =label[i][1] if label[i][1]<400 else 399
#         loss -= (torch.log(pred_start[i][label_start])+torch.log(pred_end[i][label_end])).cpu()
#     loss /= label.shape[0]
#     return loss



#Training starts
accu_counter = 0
accu_loss = 0
for batch in tqdm(range(STEP),ascii=True):

    #Batch data
    train_data= dataloader.next_train()
    while train_data.context.shape[0] != BATCH_SIZE and train_data.context.shape[1] > 300:
        train_data= dataloader.next_train()
    context = train_data.context.cuda()
    query = train_data.query.cuda()
    label = train_data.label.cuda()
    context_c = train_data.context_c.cuda()
    query_c = train_data.query_c.cuda()
    label_start = label.narrow(1,0,1).squeeze()
    label_end = label.narrow(1,1,1).squeeze()

    #Feed forward
    model.zero_grad()
    pred_start, pred_end = model(context, query, context_c, query_c)
    #Compute loss
    loss = 0
    loss = criterion(pred_start, label_start)+criterion(pred_end, label_end)

    #Backprop
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,model.parameters()), 10)
    # for name, param in model.named_parameters():
    #     if name == "output_layer.start_w.weight":
    #         startweight = param
    #     elif name == "attention_flow.alpha.weight":
    #         alphaweight = param
    #     elif name == "context_text.hidden":
    #         contexthidden = param
    # writer.add_histogram('output_layer.start_w.weight', startweight.grad.cpu().numpy(), batch,bins="doane")
    # writer.add_histogram('attention_flow.alpha.weight', alphaweight.grad.cpu().numpy(), batch,bins="doane")
    # writer.add_histogram('context_text.hidden', contexthidden.grad.cpu().numpy(), batch,bins="doane")
    optimizer.step()

    #Calculate loss
    accu_counter+=1
    accu_loss+=loss.item()
    writer.add_scalar('train_loss', loss.item(), batch)
    #Calculate period loss
    if batch % 100 == 0:
        #print([epoch,batch]," : ",accu_loss/accu_counter)
        writer.add_scalar('train_loss_avg', accu_loss/accu_counter, batch)
        print("[{}] train_loss: {}".format(batch,accu_loss/accu_counter))
        accu_counter=0
        accu_loss=0

    #Period evaluation
    if batch % 1000 == 0 and batch!=0:
        #Save model
        torch.save(model.state_dict(),"./model_checkpoint/parameters_cp_{}".format(batch))
        with torch.no_grad():
            try:
                #Validation set
                with open("./data/squad/val.context") as f:
                    ori_val_context = list(f)
                with open("./data/squad/val.answer") as f:
                    ori_val_answer = list(f)
                val_data = dataloader.next_val()
                while val_data.context.shape[0] != BATCH_SIZE:
                    val_data= dataloader.next_val()
                val_context = val_data.context.cuda()
                val_query = val_data.query.cuda()
                val_label = val_data.label.cuda()
                val_context_c = val_data.context_c.cuda()
                val_query_c = val_data.query_c.cuda()
                val_index = val_data.index.cuda()
                val_label_start = val_label.narrow(1,0,1).squeeze()
                val_label_end = val_label.narrow(1,1,1).squeeze()

                #Feed forward
                val_pred_start, val_pred_end = model(val_context, val_query, val_context_c, val_query_c)
                #loss
                val_loss = criterion(val_pred_start,val_label_start)+criterion(val_pred_end,val_label_end)
                val_predictions, val_answers = pred_answer(val_pred_start,val_pred_end, val_index, ori_val_context, ori_val_answer, BATCH_SIZE)
                with open("./predictions.txt","a+") as f:
                    f.write("val_predictions: {}\n".format(val_predictions))
                    f.write("val_answers: {}\n\n\n".format(val_answers))
                val_stat = eva.train_eval(val_answers,val_predictions)
                val_em, val_f1 = val_stat["exact_match"], val_stat["f1"]
                writer.add_scalar('val_loss', val_loss, batch)
                writer.add_scalar('val_em', val_em, batch)
                writer.add_scalar('val_f1', val_f1, batch)

                #Train set
                with open("./data/squad/train.context") as f:
                    ori_train_context = list(f)
                with open("./data/squad/train.answer") as f:
                    ori_train_answer = list(f)
                train_data = dataloader.next_train()
                while train_data.context.shape[0] != BATCH_SIZE:
                    train_data= dataloader.next_train()
                train_context = train_data.context.cuda()
                train_query = train_data.query.cuda()
                train_label = train_data.label.cuda()
                train_context_c = train_data.context_c.cuda()
                train_query_c = train_data.query_c.cuda()
                train_index = train_data.index.cuda()

                #Feed forward
                train_pred_start, train_pred_end = model(train_context, train_query, train_context_c, train_query_c)
                train_predictions, train_answers = pred_answer(train_pred_start,train_pred_end, train_index, ori_train_context, ori_train_answer, BATCH_SIZE)
                train_stat = eva.train_eval(train_answers,train_predictions)
                train_em, train_f1 = train_stat["exact_match"], train_stat["f1"]
                writer.add_scalar('train_em', train_em, batch)
                writer.add_scalar('train_f1', train_f1, batch)
            except:
                print("Failed to evaluate. But your model is saved.")
                traceback.print_exc()
