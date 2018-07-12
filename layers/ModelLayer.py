import torch
import torch.nn as nn

class ModelLayer(nn.Module):
    def __init__(self,embed_dim, batch_size, hidden_dim=100):
        super(ModelLayer,self).__init__()
        self.lstm = nn.LSTM(embed_dim,hidden_dim,bidirectional=True, batch_first=True, num_layers=2)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.hidden = self.initHidden()
        self.cell = self.initCell()

    #generate hidden cells
    def initHidden(self):
        hidden = torch.zeros(4,self.batch_size, self.hidden_dim).normal_(0.0,0.02)
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        hidden = nn.Parameter(hidden, requires_grad=True)
        return hidden

    def initCell(self):
        cell = torch.zeros(4,self.batch_size, self.hidden_dim).normal_(0.0,0.02)
        if torch.cuda.is_available():
            cell = cell.cuda()
        cell = nn.Parameter(cell, requires_grad=True)
        return cell

    def forward(self, data):
        lstm_out, last_hidden = self.lstm(data,(self.hidden,self.cell))
        return lstm_out
