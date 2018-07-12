import torch
import torch.nn as nn

from EmbeddingLayer import EmbeddingLayer
from ContextEmbLayer import ContextEmbLayer
from AttFlowLayer import AttFlowLayer
from ModelLayer import ModelLayer
from OutputLayer import OutputLayer

import time

class BiDAF(nn.Module):
    def __init__(self, vocab_vectors, char_size,
                    vector_len=100,
                    chr_emb_len=25,
                    batch_size=60):
        super(BiDAF, self).__init__()
        self.vector_len = vector_len
        self.chr_emb_len = chr_emb_len
        self.batch_size = batch_size
        self.char_size = char_size
        self.context_emb_len = self.vector_len + self.chr_emb_len
        self.embed_text = EmbeddingLayer(vocab_vectors,self.char_size,self.chr_emb_len)
        self.embed_query = EmbeddingLayer(vocab_vectors,self.char_size,self.chr_emb_len)
        self.context_text = ContextEmbLayer(self.context_emb_len,self.batch_size)
        self.context_query = ContextEmbLayer(self.context_emb_len,self.batch_size)
        self.attention_flow = AttFlowLayer(self.batch_size, self.vector_len*2)
        self.model_layer = ModelLayer(self.vector_len*8, self.batch_size)
        self.output_layer = OutputLayer(self.batch_size, self.vector_len)

    def forward(self, context, query, context_c, query_c):
        context_emb = self.embed_text(context_c,context)
        query_emb = self.embed_query(query_c, query)
        context_emb = self.context_text(context_emb)
        query_emb = self.context_text(query_emb)
        #print("Context: ",context_emb.shape)
        #print("Query: ",query_emb.shape)
        G = self.attention_flow(context_emb, query_emb)
        #print("G: ",G.shape)
        M = self.model_layer(G)
        start, end = self.output_layer(G,M)
        return start, end
