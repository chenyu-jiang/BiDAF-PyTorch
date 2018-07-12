import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputLayer(nn.Module):
    def __init__(self, batch_size, embed_length, hidden_dim=100):
        super(OutputLayer, self).__init__()
        self.embed_length = embed_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_w = nn.Linear(10*embed_length,1)
        self.lstm = nn.LSTM(2*embed_length,hidden_dim,bidirectional=True, batch_first=True)
        self.hidden = self.initHidden()
        self.cell = self.initCell()
        self.end_w = nn.Linear(10*embed_length,1)
        self.softmax = nn.Softmax(dim=1)

    def initHidden(self):
        hidden = torch.zeros(2,self.batch_size, self.hidden_dim).normal_(0.0,0.02)
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        hidden = nn.Parameter(hidden, requires_grad=True)
        return hidden

    def initCell(self):
        cell = torch.zeros(2,self.batch_size, self.hidden_dim).normal_(0.0,0.02)
        if torch.cuda.is_available():
            cell = cell.cuda()
        cell = nn.Parameter(cell, requires_grad=True)
        return cell

    def forward(self, G, M):
        GM = torch.cat((G,M),2)                             #[N,T,10d]
        M2, _ = self.lstm(M,(self.hidden,self.cell))        #[N,T,2d]
        GM2 = torch.cat((G,M2),2)                           #[N,T,10d]
        p_start = self.start_w(GM).squeeze()  #[N,T,1] -> [N,T]
        p_end = self.end_w(GM2).squeeze()     #[N,T,1] -> [N,T]
        return p_start, p_end
