import math
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X):
        #X shape is (batch_size * seq_len * hidden_dim)
        Q = self.query_proj(X)  # 我要查询什么
        K = self.key_proj(X)    # 我具备什么特征
        V = self.value_proj(X)  # 我实际包含什么信息
        
        # Q K V shape are (batch_size * seq_len * hidden*dim)
        # K^T shape is (batch_size * hidden_dim * seq_len)
        attention_value = torch.matmul(
            Q, K.transpose(-1, -2)
        )
        # attention_weight shape is (batch_size * seq * seq)
        # 代表了句子中每一个词与其他所有词的相关性打分。
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim),
            #对最后一个维度做 softmax
            dim = -1
        )
        print(attention_weight)
        # Output shape is (batch_size * seq_len * hidden_dim)
        output = torch.matmul(
            attention_weight, V
        )
        return output

# 效率优化
# 为什么这样做？
# GPU 喜欢计算大矩阵，一次大矩阵乘法比三次小矩阵乘法更能跑满 GPU 的算力。
class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # 将 Q K V 矩阵放在一起
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        
    def forward(self, X):
        # X shape is (batch_size * seq_len * hidden_dim)
        # QKV shape is (batch_size * seq_len * dim * 3)
        QKV = self.proj(X);
        Q, K, V = torch.split(QKV, self.hidden_dim, dim = -1)
        attention_weight = torch.softmax(
            torch.matmul(
                Q, K.transpose(-1, -2)
            ) / math.sqrt(self.hidden_dim),
            dim = -1
        )
        output =  attention_weight @ V
        return output

# 加入细节 
# dropout的位置
# attention mask
# output 矩阵映射
class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim: int = 728, dropout_rate = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # 将 Q K V 矩阵放在一起
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3) 
        self.attention_dropout = nn.Dropout(dropout_rate)    
        # optional
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X, attention_mask = None):
        # X shape is (batch_size * seq_len * hidden_dim)
        # QKV shape is (batch_size * seq_len * dim * 3)
        QKV = self.proj(X);
        Q, K, V = torch.split(QKV, self.hidden_dim, dim = -1)
        
        # attention_weight shape is (batch_size * seq * seq)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float('-1e20')
            )
        attention_weight = torch.softmax(attention_weight, dim = -1)
        attention_weight = self.attention_dropout(attention_weight)
        attention_result =  attention_weight @ V
        # optional (这里不懂)
        output = self.output_proj(attention_result)
        return output

class SelfAttentionFinal(nn.Module):
    def __init__(self, hidden_dim : int, dropout_rate : float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.att_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X, att_mask = None):
        # X shape : batch_size * seq_len * hidden_dim
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        
        if att_mask is not None:
            att_weight = att_weight.masked_fill(
                att_mask == 0,
                float('-inf')
            )
        att_weight = torch.softmax(att_weight, dim = -1)
        att_weight = self.att_dropout(att_weight)
        output = att_weight @ V
        return output













# X = torch.rand(3, 2, 4)
# # ()
# mask = torch.tensor(
#     [
#         [1, 1, 1, 0],
#         [1, 1, 0, 0],
#         [1, 0, 0, 0]
#     ]
# )
# self_att_net = SelfAttentionV2(4)
# print(self_att_net(X))