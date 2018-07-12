# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttFlowLayer(nn.Module):
    def __init__(self, batch_size, embed_length):
        super(AttFlowLayer,self).__init__()
        self.batch_size = batch_size
        self.embed_length = embed_length
        self.alpha = nn.Linear(3*embed_length,1,bias=False)

    def forward(self, context, query):
        #Compute similarity matrix
        shape = (self.batch_size, context.shape[1], query.shape[1], self.embed_length)
        context_extended = context.unsqueeze(2).expand(shape)                       #[N,T,2d]->[N,T,1,2d]->[N,T,J,2d]
        query_extended = query.unsqueeze(1).expand(shape)                           #[N,J,2d]->[N,1,J,2d]->[N,T,J,2d]
        multiplied = torch.mul(context_extended,query_extended)
        cated = torch.cat((context_extended,query_extended,multiplied),3)           #[h;u;h*u]
        S = self.alpha(cated).view(self.batch_size,context.shape[1],query.shape[1]) #[N,T,J,1] -> #[N,T,J]

        S_softmax_row = F.softmax(S,dim=1)              #[N,T,J]
        S_max_col, _ = torch.max(S,dim=2)               #[N,T]
        b_t = nn.functional.softmax(S_max_col,dim=1).unsqueeze(1)    #[N,T] -> [N,1,T]

        #C to Q Attention
        U_attd = torch.bmm(S_softmax_row, query)        #[N,T,J]*[N*J*2d] -> [N,T,2d]

        #Q to C Attention
        h = torch.bmm(b_t,context).squeeze()          #[N,1,T]*[N,T,2d] -> [N,1,2d] -> [N,2d]
        H_attd = h.unsqueeze(1).expand(self.batch_size, context.shape[1], self.embed_length)    #[N,2d] -> [N,1,2d] -> [N,T,2d]

        G = torch.cat((context, U_attd, context.mul(U_attd), context.mul(H_attd)),2)            #[N,T,8d]
        return G
