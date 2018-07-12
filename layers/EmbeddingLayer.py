import torch
import torch.nn as nn

from CharEmbedding import CharEmbed
from HighwayCell import HighwayCell


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_vectors, char_size, char_emb_length):
        super(EmbeddingLayer, self).__init__()
        self.char_embed = CharEmbed(char_size)
        self.word_embed = nn.Embedding.from_pretrained(vocab_vectors)
        self.highway1 = HighwayCell(char_emb_length+vocab_vectors.shape[1])
        self.dropout = nn.Dropout(0.2)
        self.highway2 = HighwayCell(char_emb_length+vocab_vectors.shape[1])
        #print("feature_size: ",char_emb_length+vocab_vectors.shape[1])
        #print("concated length:",char_emb_length+vocab_vectors.shape[1])

    def forward(self, char_data, word_data):
        char_data = self.char_embed(char_data)
        word_data = self.word_embed(word_data)
        #print("char shape:",char_data.shape)
        #print("word shape:",word_data.shape)
        data = torch.cat((char_data,word_data),2)
        #print("data shape:", data.shape)
        data = self.highway1(data)
        data = self.dropout(data)
        data = self.highway2(data)
        return data
