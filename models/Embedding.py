import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules.TransformerEncoders import *



class visualembedding(nn.Module):
    def __init__(self, level, embed_dim=512, v_inDim=2048,proj_s_drop=0.1,proj_m_drop=0.1,pos_flag='sincos',pos_dropout=0.1):
        super(visualembedding, self).__init__()
        self.level = level
        self.activ=nn.GELU()
        self.proj_s = nn.Sequential(
                        nn.Linear(v_inDim,embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_s_drop),
                        )
        self.proj_m = nn.Sequential(
                        nn.Linear(v_inDim,embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_m_drop),
                        )

        if pos_flag=='sincos':
            self.embed_scale = math.sqrt(embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim, pos_dropout)
        if pos_flag=='learned':
            self.embed_scale = 1.0
            self.pos_encoder = PositionalEncodingLearned1D(embed_dim, pos_dropout)

    def forward(self, visual_s, visual_m):

        visual_embedding_s = self.proj_s(visual_s).permute(1,2,0,3)
        visual_embedding_m = self.proj_m(visual_m).permute(1,0,2)
        visual_embedding = torch.cat([visual_embedding_s,visual_embedding_m.unsqueeze(1)],dim=1)

        visual_list = [self.pos_encoder(self.embed_scale*visual_embedding[0])] 
        for i in range(1,self.level):
            visual_list.append(self.pos_encoder(self.embed_scale*visual_embedding[np.power(2,i)-1:np.power(2,i+1)-1].reshape(-1,visual_embedding.shape[2],visual_embedding.shape[-1])))

        return visual_list


class textembedding(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=8000,wordvec_dim=300,embed_drop=0.1,last_drop=0.1):
        super(textembedding, self).__init__()
        self.activ=nn.GELU()

        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embed.weight.requires_grad = False
        self.embedding_proj = nn.Sequential(
            nn.Dropout(p=embed_drop),
            nn.Linear(wordvec_dim, embed_dim, bias=False),
            self.activ
        )
        
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim, bias=False),
            self.activ,
            nn.Dropout(p=last_drop)
        )

    def forward(self, text, text_len):

        text_embedding = self.embed(text)
        text_embedding = self.embedding_proj(text_embedding)

        self.lstm.flatten_parameters()
        text_embedding = nn.utils.rnn.pack_padded_sequence(text_embedding, text_len.cpu().numpy().tolist(), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(text_embedding)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=None)
        text_embedding = self.lstm_proj(output).transpose(0, 1)

        return text_embedding
