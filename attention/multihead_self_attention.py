import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num;
        self.head_dim = hidden_dim // head_num
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Lienar(hidden_dim, hidden_dim)
        self.o_proj = nn.Lienar(hidden_dim, hidden_dim)

        self.att_dropout = nn.Dropout(attention_dropout)
        
    def forward(self, X, attention_mask = None):
        # X = (b, s, h)
        batch_size, seq_len = X.size()
        
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # (b, s, h) -> (b, s, head_num, head_dim) -> (b, head_num, s, head_dim)
        q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k_state = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # (b, head_num, s, s)
        # k_state : (b , head_num, s, head_dim) -> (b, head_num, head_dim, s)
        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1, 2)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.att_dropout(attention_weight)
        # output_mid : (b, head_num, s, head_dim)
        output_mid = torch.matmul(
            attention_weight, v_state
        )
        # (b, head_num, s, head_dim) -> (b, s, hidden_dim)
        output_mid = output_mid.transpose(1, 2).contiguous()
        output_mid = output_mid.view(batch_size, seq_len, -1)
        
        output = self.o_proj(output_mid)
        return output
        