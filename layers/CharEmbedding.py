import torch
import torch.nn as nn
import torch.nn.functional as F

# key word arguments:
# char_emb_dim = 100, kernel_size = 5,padding_size=2
class CharEmbed(nn.Module):
    def __init__(self, char_size,**kwargs):
        super(CharEmbed,self).__init__()
        self.char_emb_dim = kwargs.get('char_emb_dim',50)
        self.kernel_size = kwargs.get('kernel_size',5)
        self.padding_size = kwargs.get('padding_size',2)
        self.pool_factor = kwargs.get('pool_factor',1)
        self.char_channel = kwargs.get("char_channel", 25)
        self.embed = nn.Embedding(char_size, self.char_emb_dim)
        self.conv = nn.Conv2d(1, self.char_channel,(self.char_emb_dim,self.kernel_size))
        self.dropout = nn.Dropout(0.2)

    def forward(self,data):
        batch_size = data.shape[0]
        data = self.dropout(self.embed(data))   # [N,SEQ,WRD,CHR]
        # [N*SEQ, CHR, WRD] -> [N*SEQ, 1, CHR, WRD]
        data = data.view(-1,self.char_emb_dim,data.size(2)).unsqueeze(1)
        # [N*SEQ, CHR_CHANNEL, 1, CONV] -> [N*SEQ, CHR_CHANNEL, CONV]
        data = self.conv(data).squeeze()
        # [N*SEQ, CHR_CHANNEL, 1] -> [N*SEQ, CHR_CHANNEL]
        data = F.max_pool1d(data,data.size(2)).squeeze()
        data = data.view(batch_size, -1, self.char_channel)
        return data
