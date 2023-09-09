"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe) # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, D) Inputs are feature embeddings.
        return x + self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1) # (B, T, D)
    
class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x

class Adapt(nn.Module):
    def __init__(self, body_dim, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, body_mean=None, body_std=None, position_encoding_length = 4096):
        super().__init__()

        self.body_dim = body_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.positional_encoding = Positional_Encoding(h_dim, position_encoding_length)
        # self.embed_timestep = Positional_Encoding(h_dim, max_timestep)
        # self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_body = torch.nn.Linear(body_dim, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        
        use_action_tanh = False # True for continuous actions

        ### prediction heads
        self.predict_body = torch.nn.Linear(h_dim, body_dim)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

        if state_mean is not None:
            self.state_mean = torch.tensor(state_mean)
            self.state_std = torch.tensor(state_std)
        
        if body_mean is not None:
            self.body_mean = torch.tensor(body_mean)
            self.body_std = torch.tensor(body_std)


    def forward(self, states, actions, bodies):
        
        B, T, _ = states.shape

        state_embeddings = self.positional_encoding(self.embed_state(states))
        action_embeddings = self.positional_encoding(self.embed_action(actions))
        body_embeddings = self.positional_encoding(self.embed_body(bodies))

        # stack rtg, states and actions and reshape sequence as
        # (s_0, b_0, a_0, s_1, b_1, a_1, s_2, b_2, a_2 ...)
        # inference:s->b->a

        h = torch.stack(
            (state_embeddings, body_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)


        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence s_0, b_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence s_0, b_0, a_0 ... s_t, b_t, 
        # h[:, 2, t] is conditioned on the input sequence s_0, b_0, a_0 ... s_t, b_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (s_t, b_t, a_t) in sequence.

        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions

        state_preds = self.predict_state(h[:,2])    # predict next state given s, b, a
        action_preds = self.predict_action(h[:,1])  # predict action given s, b
        body_preds = self.predict_body(h[:,0])     # predict next body given s
        
        return state_preds, action_preds, body_preds
    