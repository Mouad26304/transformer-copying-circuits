import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model),
                                 nn.GELU(),
                                 nn.Linear(4*d_model,d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self, x):
        b,s,d = x.size()
        residual = x
        x = self.ln1(x)
        Q,V,K = self.W_q(x), self.W_v(x), self.W_k(x)
        Q,V,K = Q.reshape(b,s,self.num_heads,self.d_head),V.reshape(b,s,self.num_heads,self.d_head),K.reshape(b,s,self.num_heads,self.d_head)
        K=K.transpose(1,2)
        Q=Q.transpose(1,2)
        V=V.transpose(1,2)
        scores = Q@K.transpose(-1,-2)/math.sqrt(self.d_head)
        mask = torch.triu( torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -math.inf)
        scores =torch.softmax(scores, dim = -1)
        scores = scores@V
        scores = scores.transpose(1,2)
        scores = scores.reshape(b,s,d)
        output = self.W_o(scores)
        x= residual+output
        residual = x
        x=self.ln2(x)
        x_mlp = self.mlp(x)
        x=residual+x_mlp
        return x

class AssociativeRecallTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len,d_model,num_heads, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(d_model,num_heads) for _ in range(num_layers)])
        self.token_projection = nn.Linear(vocab_size,d_model, bias = False)
        self.position_projection = nn.Linear(seq_len,d_model, bias = False)
        self.lm_head = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        x = F.one_hot(x, num_classes=self.token_projection.in_features).float()
        positions = torch.arange(0,x.size(1), device=x.device)
        positions = F.one_hot(positions, self.position_projection.in_features).float()
        positions = self.position_projection(positions)
        x= self.token_projection(x)
        x+=positions
        for block in self.blocks:
             x = block(x)
        return self.lm_head(x)
