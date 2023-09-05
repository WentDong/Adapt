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
        # # x: (B, T) Inputs are the index of steps.
        # return torch.gather(self.pe.repeat(x.size(0), 1, 1), 1, x.unsqueeze(-1).repeat(1,1,self.d_model)) # B x T x D
    
class Positional_Encoding_On_Step(nn.Module): #Version for 1D input
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
        # # x: (B, T, D) Inputs are feature embeddings.
        # return x + self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1) # (B, T, D)
        # x: (B, T) Inputs are the index of steps.
        return torch.gather(self.pe.repeat(x.size(0), 1, 1), 1, x.unsqueeze(-1).repeat(1,1,self.d_model)) # B x T x D
     
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


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, use_rtg=True):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep
        self.use_rtg = use_rtg

        if not use_rtg:
            print("Return To Go is not used")

        ### transformer blocks
        input_seq_len = 3 * context_len if self.use_rtg else 2 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        # self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)


    def forward(self, timesteps, states, actions):
        # def forward(self, timesteps, states, actions, returns_to_go=None, body=None):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        assert not self.use_rtg

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        if self.use_rtg:
            h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        else:
            h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = None
        state_preds = self.predict_state(h[:,1])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,0])  # predict action given r, s

        return state_preds, action_preds, return_preds


class LeggedTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None):
        super().__init__()

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
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)


    def forward(self, timesteps, states, actions, leg_length):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(leg_length) + time_embeddings

        # stack states, body and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)

        h = torch.stack(
            (state_embeddings, returns_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)


        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t, b_t, 
        # h[:, 2, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t, b_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (s_t, b_t, a_t) in sequence.

        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions

        body_preds = self.predict_rtg(h[:,0])     # predict next rtg given s, b, a
        state_preds = self.predict_state(h[:,2])    # predict next state given s, b, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s
        
        return state_preds, action_preds, body_preds

class LeggedTransformerDistill(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, position_encoding_length=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep

        ### transformer blocks
        input_seq_len = 2 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.positional_encoding = Positional_Encoding(h_dim, position_encoding_length)
        # self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = False # True for continuous actions

        ### prediction heads
        # self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)


    def forward(self, states, actions, bodies= None):

        B, T, _ = states.shape


        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.positional_encoding(self.embed_state(states))
        action_embeddings = self.positional_encoding(self.embed_action(actions))
        # returns_embeddings = self.embed_rtg(leg_length) + time_embeddings

        # stack states, body and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)

        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)


        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t, b_t, 
        # h[:, 2, t] is conditioned on the input sequence b_0, s_0, a_0 ... s_t, b_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (s_t, b_t, a_t) in sequence.

        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(h[:,1])    # predict next state given s, b, a
        action_preds = self.predict_action(h[:,0])  # predict action given r, s
        
        return state_preds, action_preds, None

class LeggedTransformerPro(nn.Module):
    def __init__(self, body_dim, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, body_mean=None, body_std=None):
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
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_body = torch.nn.Linear(body_dim, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        
        #! 此处根据余琛学长意见去掉最后一层的action，原本这一层会让他的机器狗在实机上表现得更好 又改成true试一下
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


    def forward(self, timesteps, states, actions, body):
        
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        # pdb.set_trace()
        body_embeddings = self.embed_body(body) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)

        h = torch.stack(
            (body_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)


        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.

        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions

        return_preds = self.predict_body(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s
        
        return state_preds, action_preds, return_preds

class LeggedTransformerBody(nn.Module):
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
        
        #! 此处根据余琛学长意见去掉最后一层的action，原本这一层会让他的机器狗在实机上表现得更好 又改成true试一下
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

        # time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.positional_encoding(self.embed_state(states))
        action_embeddings = self.positional_encoding(self.embed_action(actions))
        # pdb.set_trace()
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
    
class LeggedTransformerBody_Global_Steps(nn.Module):
    def __init__(self, body_dim, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, body_mean=None, body_std=None):
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
        # self.positional_encoding = Positional_Encoding(h_dim, max_timestep)
        self.embed_timestep = Positional_Encoding_On_Step(h_dim, max_timestep)
        # self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_body = torch.nn.Linear(body_dim, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        
        #! 此处根据余琛学长意见去掉最后一层的action，原本这一层会让他的机器狗在实机上表现得更好 又改成true试一下
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


    def forward(self, timesteps, states, actions, bodies):
        
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        # pdb.set_trace()
        body_embeddings = self.embed_body(bodies) + time_embeddings

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
    
class MLPBCModel(nn.Module):
    
    """
    Simple MLP that predicts next action a from past states s.

    self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, use_rtg=True
    """

    def __init__(self, body_dim, state_dim, act_dim, n_blocks, h_dim, context_len=1, drop_p=0.1, state_mean=None, state_std=None):
        super().__init__()

        self.hidden_size = h_dim
        self.max_length = context_len
        self.body_dim = body_dim
        self.state_dim = state_dim
        self.act_dim = act_dim

        layers = [nn.Linear(self.max_length*(self.state_dim+self.body_dim), self.hidden_size)]
        for _ in range(n_blocks-1):
            layers.extend([
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
                # nn.Dropout(drop_p),
                nn.Linear(self.hidden_size, self.hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            # nn.Dropout(drop_p),
            nn.Linear(self.hidden_size, self.act_dim),
            # nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)

    def forward(self, states, actions= None, bodies=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1).to(dtype=torch.float32)  # concat states
        bodies = bodies[:,-self.max_length:].reshape(bodies.shape[0], -1).to(dtype=torch.float32)

        actions = self.model(torch.cat((states, bodies), dim=1)).reshape(states.shape[0], -1, self.act_dim)

        return None, actions, None

    def get_action(self, states, bodies, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
            bodies = torch.cat(
                [torch.zeros((1, self.max_length-bodies.shape[1], self.body_dim),
                             dtype=torch.float32, device=bodies.device), bodies], dim=1
            ).to(dtype=torch.float32)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, bodies)
        return actions[0,-1]

    def get_action_batch(self, states, bodies):
        # states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            # states = torch.cat(
            #     [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
            #                  dtype=torch.float32, device=states.device), states], dim=1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), 
                             dtype=torch.float32, device=states.device), states], dim=1)
            bodies = torch.cat(
                [torch.zeros((bodies.shape[0], self.max_length-bodies.shape[1], self.body_dim),
                             dtype=torch.float32, device=bodies.device), bodies], dim=1
            ).to(dtype=torch.float32)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, bodies)
        return actions[:,-1]
    

    
class MLPBCModel_Distill(nn.Module):
    
    """
    Simple MLP that predicts next action a from past states s.

    self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None, use_rtg=True
    """

    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len=1, drop_p=0.1, state_mean=None, state_std=None):
        super().__init__()

        self.hidden_size = h_dim
        self.max_length = context_len
        self.state_dim = state_dim
        self.act_dim = act_dim

        layers = [nn.Linear(self.max_length*(self.state_dim), self.hidden_size)]
        for _ in range(n_blocks-1):
            layers.extend([
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
                # nn.Dropout(drop_p),
                nn.Linear(self.hidden_size, self.hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            # nn.Dropout(drop_p),
            nn.Linear(self.hidden_size, self.act_dim),
            # nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

        self.state_mean = torch.tensor(state_mean)
        self.state_std = torch.tensor(state_std)

    def forward(self, states, actions= None, bodies=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1).to(dtype=torch.float32)  # concat states

        actions = self.model(states).reshape(states.shape[0], -1, self.act_dim)

        return None, actions, None

    def get_action(self, states, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None)
        return actions[0,-1]

    def get_action_batch(self, states):
        # states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            # states = torch.cat(
            #     [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
            #                  dtype=torch.float32, device=states.device), states], dim=1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), 
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None)
        return actions[:,-1]
    


class Model_Prediction(nn.Module):
    def __init__(self, state_dim, body_dim, context_len=100, n_blocks=3, h_dim=128, 
                 n_heads=2, drop_p=0.1, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.body_dim = body_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep

        ### transformer blocks
        input_seq_len =  (context_len) * 2
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.position_encoding = Positional_Encoding(h_dim, max_timestep)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_action = torch.nn.Linear(body_dim, h_dim)

        ### prediction heads
        # self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.pred_body = nn.Linear(h_dim, body_dim)
        self.cls_body = nn.Linear(h_dim, body_dim)


    def forward(self, states, actions):
        # def forward(self, timesteps, states, actions, returns_to_go=None, body=None):

        B, T, _ = states.shape


        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.position_encoding(self.embed_state(states))
        action_embeddings = self.position_encoding(self.embed_action(actions))
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)
        # h = state_embeddings

        h = self.embed_ln(h)
        # transformer and prediction
        h = self.transformer(h)

        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)
        # get predictions
        body_preds = self.pred_body(h[:,1])  
        body_cls = nn.Softmax(self.body_dim)(self.cls_body(h[:,0]))
        return body_preds
    
    def take_body(self, timesteps, states, actions):
        return torch.clamp(self.forward(timesteps, states, actions),min=0, max=1)
