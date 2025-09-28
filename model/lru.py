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
        cat_emb = torch.load(f'./data/{args.dataset_code}/cat.pt').float()
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb).to(args.device)
        self.cat_linear = nn.Linear(2 * args.bert_hidden_units, cat_emb.shape[-1])
        txt_emb = torch.load(f'./data/{args.dataset_code}/txt_embeddings.pt').float()
        self.txt_embedding = nn.Embedding.from_pretrained(txt_emb).to(args.device)
        self.txt_linear = nn.Linear(txt_emb.shape[-1], args.bert_hidden_units)

    def get_category_embedding(self):
        return self.cat_embedding

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
        x, mask, pe = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)

class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)
        self.positional_embedding = nn.Embedding(vocab_size, embed_size)

    def get_mask(self, x):
        return (x > 0)
    
    def forward(self, x):
        mask = self.get_mask(x)                   
        positional_ids = torch.cumsum(mask, dim=1)   
        positional_ids = positional_ids * mask
        pos_emb = self.positional_embedding(positional_ids)      
        x = self.token(x) + pos_emb
        return self.layer_norm(self.embed_dropout(x)), mask, pos_emb

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
    
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features * 2)
        self.linear2 = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        hidden_states = self.linear1(x)
        gate, activated = hidden_states.chunk(2, dim=-1)
        activated = F.silu(activated)
        output = self.linear2(gate * activated)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff * 2)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_proj = self.w_1(x)  # [B, L, d_ff*2]
        gate, act = x_proj.chunk(2, dim=-1)
        act = F.silu(act)
        x_ = self.dropout(gate * act)
        x_ = self.dropout(self.w_2(x_))
        return self.layer_norm(x_ + x)