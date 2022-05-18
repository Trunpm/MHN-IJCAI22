import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Function
from torch.nn import functional as F
from .modules.TransformerEncoders import *


class ADA(nn.Module):
    def __init__(self, embed_dim):
        super(ADA, self).__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5
        self.activ=nn.GELU()

        self.proj1 = nn.Sequential(nn.Linear(embed_dim,embed_dim//2), self.activ)
        self.proj2 = nn.Sequential(nn.Linear(embed_dim,embed_dim//2), self.activ)

    def forward(self, v_1, v_2):
  
        v_1 = v_1.transpose(0,1)
        v_2 = v_2.transpose(0,1)
        attn = F.softmax(torch.bmm(self.proj1(v_1),self.proj2(v_2).transpose(1,2))*self.scaling,dim=-1) #Batch X1 X2
        v_1 = v_1 + torch.bmm(attn,v_2)
        return v_1.transpose(0,1)

class PyTransformer(nn.Module):
    def __init__(self, level, embed_dim, num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1):
        super(PyTransformer, self).__init__()
        self.level = level
        self.scaling = embed_dim ** -0.5
        self.activ=nn.GELU()

        self.CrossmodalEncoder = TransformerEncoder_nopos_v2(embed_dim,num_heads,attn_dropout,res_dropout,activ_dropout,1)
        self.Up_ = nn.ModuleList([])
        for i in range(self.level-1):
            self.Up_.append(ADA(embed_dim))
        
        self.TransformerEncoder = TransformerEncoder_nopos_v1(embed_dim,num_heads,attn_dropout,res_dropout,activ_dropout,1)
        
    def forward(self, visual_list, text_embedding, text_len):
  
        v_media = []
        for i in range(self.level):
            visual_list[i],text_embedding = self.CrossmodalEncoder(visual_list[i],text_embedding,text_embedding,text_len)
            v_media.append(visual_list[i].clone())
            if i<self.level-1:
                visual_list[i+1] = self.Up_[i-1](visual_list[i+1], visual_list[i])

        for i in range(self.level):
            v_media[i] = self.TransformerEncoder(v_media[i],v_media[i],v_media[i])

        visual_out = torch.stack([item.mean(dim=0) for item in v_media],dim=1)
        text_out = torch.stack([text_embedding[0:text_len[j],j,:].mean(dim=0) for j in range(text_len.shape[0])],dim=0).unsqueeze(-1)
        Out = (visual_out*F.softmax(torch.bmm(visual_out,text_out)*self.scaling,dim=1)).sum(dim=1)

        return Out