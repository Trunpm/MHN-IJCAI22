import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class PositionalEncodingLearned1D(nn.Module):
    """
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncodingLearned1D(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLearned1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        idx = torch.arange(x.shape[0],device=x.device)
        x = x + self.pos_embed(idx).unsqueeze(1)
        return self.dropout(x)


        


class TransformerEncoderLayer_QKV_v1(nn.Module):
    """
    Completed TransformerEncoderLayer for input Q K V

    """
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu'):
        super(TransformerEncoderLayer_QKV_v1,self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout)
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc2 = nn.Linear(4*embed_dim, embed_dim)
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).bool()

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
  
    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len,  batch, embed_dim)
        """
        residual = x_q
        x_q = self.maybe_layer_norm(0, x_q, before=True)
        if x_k is None:
            key_padding_mask = self._get_key_padding_mask(x_q.shape[0],kv_len) if kv_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_q, value=x_q, key_padding_mask=key_padding_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            key_padding_mask = self._get_key_padding_mask(x_k.shape[0],kv_len) if kv_len is not None else None
            if x_v is None:
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_k, key_padding_mask=key_padding_mask)
            else:
                x_v = self.maybe_layer_norm(0, x_v, before=True)
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_v, key_padding_mask=key_padding_mask)
             
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(0, x_q, after=True)

        residual = x_q
        x_q = self.maybe_layer_norm(1, x_q, before=True)
        x_q = self.activ(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.activ_dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(1, x_q, after=True)
        return x_q




class TransformerEncoder_nopos_v1(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu', num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV_v1(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                activ_dropout=activ_dropout,
                                                activation=activation)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, x_v, kv_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        return x_q


















class TransformerEncoderLayer_QKV_v2(nn.Module):
    """
    Completed TransformerEncoderLayer for input Q K V

    """
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu'):
        super(TransformerEncoderLayer_QKV_v2,self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout)
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc2 = nn.Linear(4*embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc4 = nn.Linear(4*embed_dim, embed_dim)
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).bool()

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
  
    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len,  batch, embed_dim)
        """
        residual = x_q
        x_k_ = x_k
        x_q = self.maybe_layer_norm(0, x_q, before=True)
        if x_k is None:
            key_padding_mask = self._get_key_padding_mask(x_q.shape[0],kv_len) if kv_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_q, value=x_q, key_padding_mask=key_padding_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            key_padding_mask = self._get_key_padding_mask(x_k.shape[0],kv_len) if kv_len is not None else None
            if x_v is None:
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_k, key_padding_mask=key_padding_mask)
            else:
                x_v = self.maybe_layer_norm(0, x_v, before=True)
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_v, key_padding_mask=key_padding_mask)
             
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(0, x_q, after=True)

        residual = x_q
        x_q = self.maybe_layer_norm(1, x_q, before=True)
        x_q = self.activ(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.activ_dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(1, x_q, after=True)

        residual_k = x_k_
        x_k_ = self.maybe_layer_norm(1, x_k_, before=True)
        x_k_ = self.activ(self.fc3(x_k_))
        x_k_ = F.dropout(x_k_, p=self.activ_dropout, training=self.training)
        x_k_ = self.fc4(x_k_)
        x_k_ = F.dropout(x_k_, p=self.res_dropout, training=self.training)
        x_k_ = residual_k + x_k_
        x_k_ = self.maybe_layer_norm(1, x_k_, after=True)

        return x_q,x_k_


class TransformerEncoder_nopos_v2(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu', num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV_v2(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                activ_dropout=activ_dropout,
                                                activation=activation)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q,x_k = layer(x_q, x_k, x_v, kv_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        x_k = self.layer_norm(x_k)
        return x_q,x_k