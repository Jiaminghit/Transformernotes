import math
import torch
import torch.nn as nn

class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout_rate = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        # layer func
        # multi - head - att
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(attention_dropout_rate)
        self.att_ln = nn.LayerNorm(hidden_dim, eps = 0.0000001)
        
        # feed - forward - net
        self.up_proj = nn.Lienar(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.drop_ffn = nn.Dropout(0.1)
        self.ffn_ln = nn.LayerNorm(hidden_dim, eps = 0.0000001)
    
    def attention_layer(self, query, key, value, attention_mask = None):
        # key : (b, head_num, s, head_dim) -> (b, head_num, head_dim, s)
        key = key.transpose(2, 3)
        attention_weight = torch.matmul(
            query, key
        ) / math.sqrt(self.head_dim)
        # ?????
        if attention_mask is not None:
            attention_mask = attention_mask.tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        else:
            attention_mask = torch.ones_like(attention_weight).tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        attention_weight = torch.softmax(attention_weight, dim = -1)
        attention_weight = self.drop_att(attention_weight)
        
        mid_out = torch.matmul(attention_weight, value)
        # (b, head_num, s, head_dim) -> (b, s, head_num, head_dim)
        mid_out.transpose(1, 2).contiguous()
        batch, seq, _, _ = mid_out.size()
        mid_out = mid_out.view(batch, seq, -1)
        
        output = self.o_proj(mid_out)
        
        return output
    
    def mma(self, X, mask = None):
        # (b, s, h) -> (b, s, head_num, head_dim) -> (b, head_num, s, head_dim)
        batch, seq, _ = X.size()
        query = self.q_proj(X).view(batch, seq, self.head_num, -1).transpose(1, 2)
        key = self.k_proj(X).view(batch, seq, self.head_num, -1).transpose(1, 2)
        value = self.v_proj(X).view(batch, seq, self.head_num, -1).transpose(1, 2)
        self.attention_layer(X, query, key, value, mask)
        return X
    
    def ffn(self, X):
        up = self.up_proj(X)
        up = self.act_fn(up)
        down = self.down_proj(up)
        down = self.drop_ffn(down)
        return self.ffn_ln(X + down)
    
    def forward(self, X, attention_mask = None):
        X = self.mma(X, attention_mask)
        X = self.ffn(X)
        return X
        
