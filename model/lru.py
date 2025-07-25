import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = LRUEmbedding(self.args)
        self.model = LRUModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, x, labels=None):
        x, mask = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)

class RoPE(nn.Module):
    def __init__(self, seq_len, d_model, base=10000):
        super().__init__()

        k_max = d_model // 2
        theta = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.outer(torch.arange(seq_len), theta)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)  # [seq_len, k_max, 2]
        rotations = rotations.reshape(seq_len, -1)  # [seq_len, d_model]
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        batch_size, seq_len, d_model = x.shape
        rotations = self.rotations[:seq_len]  # [seq_len, d_model]
        rotations = rotations.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotations_complex = torch.view_as_complex(rotations.reshape(*rotations.shape[:-1], -1, 2))
        pe_x = rotations_complex * x_complex
        return torch.view_as_real(pe_x).reshape(batch_size, seq_len, d_model)

# class LRUEmbedding(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         vocab_size = args.num_items + 1
#         embed_size = args.bert_hidden_units
        
#         self.token = nn.Embedding(vocab_size, embed_size)
#         # self.positional_embedding = nn.Embedding(vocab_size, embed_size)
#         self.rope = RoPE(vocab_size, embed_size)
#         self.layer_norm = nn.LayerNorm(embed_size)
#         self.embed_dropout = nn.Dropout(args.bert_dropout)

#     def get_mask(self, x):
#         return (x > 0)

#     def forward(self, x):
#         mask = self.get_mask(x)
#         x = self.token(x) + self.rope(x)    
#         return self.layer_norm(self.embed_dropout(x)), mask

class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask

class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        # self.hidden_size =20
        layers = args.bert_num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)
        
        # prediction layer
        if self.args.dataset_code != 'xlong':
            scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
            return scores, x
        else:
            assert labels is not None
            if self.training:
                num_samples = self.args.negative_sample_size  # 100
                samples = torch.randint(1, self.args.num_items+1, size=(*x.shape[:2], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size  # 10000
                samples = torch.randint(1, self.args.num_items+1, size=(x.shape[0], num_samples,))  # only one time step
                all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_.reshape(labels.shape)
            

class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.bert_hidden_units
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=args.bert_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        # self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)
    

### LRU PE
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np


# class LRU(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.embedding = LRUEmbedding(self.args)
#         self.model = LRUModel(self.args)
#         self.truncated_normal_init()

#     def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
#         with torch.no_grad():
#             l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
#             u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

#             for n, p in self.named_parameters():
#                 if not 'layer_norm' in n and 'params_log' not in n:
#                     if torch.is_complex(p):
#                         p.real.uniform_(2 * l - 1, 2 * u - 1)
#                         p.imag.uniform_(2 * l - 1, 2 * u - 1)
#                         p.real.erfinv_()
#                         p.imag.erfinv_()
#                         p.real.mul_(std * math.sqrt(2.))
#                         p.imag.mul_(std * math.sqrt(2.))
#                         p.real.add_(mean)
#                         p.imag.add_(mean)
#                     else:
#                         p.uniform_(2 * l - 1, 2 * u - 1)
#                         p.erfinv_()
#                         p.mul_(std * math.sqrt(2.))
#                         p.add_(mean)

#     def forward(self, x, labels=None):
#         x, mask = self.embedding(x)
#         return self.model(x, self.embedding.token.weight, mask, labels=labels)


# class LRUEmbedding(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         vocab_size = args.num_items + 1
#         embed_size = args.bert_hidden_units
        
#         self.token = nn.Embedding(vocab_size, embed_size)
#         self.layer_norm = nn.LayerNorm(embed_size)
#         self.embed_dropout = nn.Dropout(args.bert_dropout)
#         self.positional_embedding = nn.Embedding(vocab_size, embed_size)

#     def get_mask(self, x):
#         return (x > 0)

#     def forward(self, x):
#         mask = self.get_mask(x)
#         seq_len = x.size(1)
#         position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0) 
#         position_emb = self.positional_embedding(position_ids)
#         x = self.token(x) + position_emb
#         return self.layer_norm(self.embed_dropout(x)), mask


# class LRUModel(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.bert_hidden_units
#         layers = args.bert_num_blocks

#         self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
#         self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

#     def forward(self, x, embedding_weight, mask, labels=None):
#         # left padding to the power of 2
#         seq_len = x.size(1)
#         log2_L = int(np.ceil(np.log2(seq_len)))
#         x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
#         mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

#         # LRU blocks with pffn
#         for lru_block in self.lru_blocks:
#             x = lru_block.forward(x, mask_)
#         x = x[:, -seq_len:]  # B x L x D (64)
        
#         # prediction layer
#         if self.args.dataset_code != 'xlong':
#             scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
#             return scores, None
#         else:
#             assert labels is not None
#             if self.training:
#                 num_samples = self.args.negative_sample_size  # 100
#                 samples = torch.randint(1, self.args.num_items+1, size=(*x.shape[:2], num_samples,))
#                 all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
#                 sampled_embeddings = embedding_weight[all_items]
#                 scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
#                 labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
#                 return scores, labels_
#             else:
#                 num_samples = self.args.xlong_negative_sample_size  # 10000
#                 samples = torch.randint(1, self.args.num_items+1, size=(x.shape[0], num_samples,))  # only one time step
#                 all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
#                 sampled_embeddings = embedding_weight[all_items]
#                 scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
#                 labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
#                 return scores, labels_.reshape(labels.shape)
            

# class LRUBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         hidden_size = args.bert_hidden_units
#         self.lru_layer = LRULayer(
#             d_model=hidden_size, dropout=args.bert_attn_dropout)
#         self.feed_forward = PositionwiseFeedForward(
#             d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
#     def forward(self, x, mask):
#         x = self.lru_layer(x, mask)
#         x = self.feed_forward(x)
#         return x
    

# class LRULayer(nn.Module):
#     def __init__(self,
#                  d_model,
#                  dropout=0.1,
#                  use_bias=True,
#                  r_min=0.8,
#                  r_max=0.99):
#         super().__init__()
#         self.embed_size = d_model
#         self.hidden_size = 2 * d_model
#         self.use_bias = use_bias

#         # init nu, theta, gamma
#         u1 = torch.rand(self.hidden_size)
#         u2 = torch.rand(self.hidden_size)
#         nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
#         theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
#         diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
#         gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
#         self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

#         # Init B, C, D
#         self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
#         self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
#         # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
#         self.out_vector = nn.Identity()
        
#         # Dropout and layer norm
#         self.dropout = nn.Dropout(p=dropout)
#         self.layer_norm = nn.LayerNorm(self.embed_size)

#     def lru_parallel(self, i, h, lamb, mask, B, L, D):
#         # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
#         # The original implementation is slightly slower and does not consider 0 padding
#         l = 2 ** i
#         h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
#         mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
#         h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

#         if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
#         h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
#         h = torch.cat([h1, h2], axis=1)
#         return h, lamb

#     def forward(self, x, mask):
#         # compute bu and lambda
#         nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
#         lamb = torch.exp(torch.complex(-nu, theta))
#         h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
#         # compute h in parallel
#         log2_L = int(np.ceil(np.log2(h.size(1))))
#         B, L, D = h.size(0), h.size(1), h.size(2)
#         for i in range(log2_L):
#             h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
#         x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
#         return self.layer_norm(x)  # residual connection introduced above 
    

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         x_ = self.dropout(self.activation(self.w_1(x)))
#         return self.layer_norm(self.dropout(self.w_2(x_)) + x)

### NEW ROPE 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np


# class LRU(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.embedding = LRUEmbedding(self.args)
#         self.model = LRUModel(self.args)
#         self.truncated_normal_init()

#     def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
#         with torch.no_grad():
#             l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
#             u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

#             for n, p in self.named_parameters():
#                 if not 'layer_norm' in n and 'params_log' not in n:
#                     if torch.is_complex(p):
#                         p.real.uniform_(2 * l - 1, 2 * u - 1)
#                         p.imag.uniform_(2 * l - 1, 2 * u - 1)
#                         p.real.erfinv_()
#                         p.imag.erfinv_()
#                         p.real.mul_(std * math.sqrt(2.))
#                         p.imag.mul_(std * math.sqrt(2.))
#                         p.real.add_(mean)
#                         p.imag.add_(mean)
#                     else:
#                         p.uniform_(2 * l - 1, 2 * u - 1)
#                         p.erfinv_()
#                         p.mul_(std * math.sqrt(2.))
#                         p.add_(mean)

#     def forward(self, x, labels=None):
#         x, mask = self.embedding(x)
#         return self.model(x, self.embedding.token.weight, mask, labels=labels)

# class RoPE(nn.Module):
#     def __init__(self, seq_len, d_model, base=10000):
#         super().__init__()
#         assert d_model % 2 == 0, "d_model must be even for RoPE complex representation"
#         half_dim = d_model // 2

#         # Compute thetas
#         theta = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))  # [half_dim]
#         position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)                # [seq_len, 1]
#         angles = position * theta                                                            # [seq_len, half_dim]

#         # Encode with sin/cos
#         cos = torch.cos(angles)  # [seq_len, half_dim]
#         sin = torch.sin(angles)  # [seq_len, half_dim]

#         # Register as buffer
#         self.register_buffer("cos", cos)
#         self.register_buffer("sin", sin)

#     def forward(self, x):
#         """
#         x: [batch_size, seq_len, d_model] where d_model % 2 == 0
#         returns: same shape with RoPE positional encoding applied
#         """
#         B, L, D = x.shape
#         assert D % 2 == 0, "d_model must be even"
#         half_dim = D // 2

#         x1 = x[:, :, :half_dim]
#         x2 = x[:, :, half_dim:]

#         # Get positional embeddings
#         cos = self.cos[:L].unsqueeze(0).to(x.device)  # [1, L, half_dim]
#         sin = self.sin[:L].unsqueeze(0).to(x.device)  # [1, L, half_dim]

#         # RoPE rotation
#         x_rotated = torch.cat([
#             x1 * cos - x2 * sin,
#             x1 * sin + x2 * cos
#         ], dim=-1)

#         return x_rotated

# class LRUEmbedding(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         vocab_size = args.num_items + 1
#         embed_size = args.bert_hidden_units
        
#         self.token = nn.Embedding(vocab_size, embed_size)
#         # self.positional_embedding = nn.Embedding(vocab_size, embed_size)
#         self.rope = RoPE(vocab_size, embed_size)
#         self.layer_norm = nn.LayerNorm(embed_size)
#         self.embed_dropout = nn.Dropout(args.bert_dropout)

#     def get_mask(self, x):
#         return (x > 0)

#     def forward(self, x):
#         mask = self.get_mask(x)
#         tok_embed = self.token(x)
#         pos_embed = self.rope(tok_embed)
#         x = tok_embed + pos_embed  # có thể chỉ dùng pos_embed cũng được nếu đã tính trong RoPE
#         return self.layer_norm(self.embed_dropout(x)), mask

# class LRUModel(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.bert_hidden_units
#         # self.hidden_size =20
#         layers = args.bert_num_blocks

#         self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
#         self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

#     def forward(self, x, embedding_weight, mask, labels=None):
#         # left padding to the power of 2
#         seq_len = x.size(1)
#         log2_L = int(np.ceil(np.log2(seq_len)))
#         x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
#         mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

#         # LRU blocks with pffn
#         for lru_block in self.lru_blocks:
#             x = lru_block.forward(x, mask_)
#         x = x[:, -seq_len:]  # B x L x D (64)
        
#         # prediction layer
#         if self.args.dataset_code != 'xlong':
#             scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
#             return scores, None
#         else:
#             assert labels is not None
#             if self.training:
#                 num_samples = self.args.negative_sample_size  # 100
#                 samples = torch.randint(1, self.args.num_items+1, size=(*x.shape[:2], num_samples,))
#                 all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
#                 sampled_embeddings = embedding_weight[all_items]
#                 scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
#                 labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
#                 return scores, labels_
#             else:
#                 num_samples = self.args.xlong_negative_sample_size  # 10000
#                 samples = torch.randint(1, self.args.num_items+1, size=(x.shape[0], num_samples,))  # only one time step
#                 all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
#                 sampled_embeddings = embedding_weight[all_items]
#                 scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
#                 labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
#                 return scores, labels_.reshape(labels.shape)
            

# class LRUBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         hidden_size = args.bert_hidden_units
#         self.lru_layer = LRULayer(
#             d_model=hidden_size, dropout=args.bert_attn_dropout)
#         self.feed_forward = PositionwiseFeedForward(
#             d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
#     def forward(self, x, mask):
#         x = self.lru_layer(x, mask)
#         x = self.feed_forward(x)
#         return x
    

# class LRULayer(nn.Module):
#     def __init__(self,
#                  d_model,
#                  dropout=0.1,
#                  use_bias=True,
#                  r_min=0.8,
#                  r_max=0.99):
#         super().__init__()
#         self.embed_size = d_model
#         self.hidden_size = 2 * d_model
#         self.use_bias = use_bias

#         # init nu, theta, gamma
#         u1 = torch.rand(self.hidden_size)
#         u2 = torch.rand(self.hidden_size)
#         nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
#         theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
#         diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
#         gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
#         self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

#         # Init B, C, D
#         self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
#         self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
#         # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
#         self.out_vector = nn.Identity()
        
#         # Dropout and layer norm
#         self.dropout = nn.Dropout(p=dropout)
#         self.layer_norm = nn.LayerNorm(self.embed_size)

#     def lru_parallel(self, i, h, lamb, mask, B, L, D):
#         # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
#         # The original implementation is slightly slower and does not consider 0 padding
#         l = 2 ** i
#         h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
#         mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
#         h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

#         if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
#         h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
#         h = torch.cat([h1, h2], axis=1)
#         return h, lamb

#     def forward(self, x, mask):
#         # compute bu and lambda
#         nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
#         lamb = torch.exp(torch.complex(-nu, theta))
#         h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
#         # compute h in parallel
#         log2_L = int(np.ceil(np.log2(h.size(1))))
#         B, L, D = h.size(0), h.size(1), h.size(2)
#         for i in range(log2_L):
#             h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
#         x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
#         return self.layer_norm(x)  # residual connection introduced above 
    

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         x_ = self.dropout(self.activation(self.w_1(x)))
#         return self.layer_norm(self.dropout(self.w_2(x_)) + x)