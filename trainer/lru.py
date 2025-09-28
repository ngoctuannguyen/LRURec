from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.bce = nn.BCEWithLogitsLoss()

    def calculate_loss(self, batch):
        seqs, labels = batch
        
        if self.args.dataset_code != 'xlong':
            logits, hidden_items = self.model(seqs)
            # print("Hidden items", hidden_items)
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
            ##### 0.3 is a weight factor of loss function
            loss = self.ce(logits, labels) + \
                  args.CP_loss_weight * self.CP(seqs)
        else:
            logits, labels_ = self.model(seqs, labels=labels)
            logits = logits.reshape(-1, logits.size(-1))
            labels_[labels==0] = 0
            labels_ = labels_.view(-1)
            loss = self.ce(logits, labels_)
        return loss 
    
    def CP(self, input, padding_idx=0):
        item_list = input
        nonzero_idx = torch.where(input != padding_idx)
        item_emb, pos_emb = self.model.embedding(item_list)[0], self.model.embedding(item_list)[2]
        txt_emb = self.model.txt_embedding(item_list)
        txt_emb = self.model.txt_linear(txt_emb) + pos_emb
        item_attribute_score = self.model.cat_linear(torch.cat([item_emb, txt_emb], dim=-1))
        item_attribute_target = self.model.cat_embedding(item_list)
        attr_loss = self.bce(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss
    
    def IDCL(self, seqs, logits):
        # logits: [batch_size, seq_len, embed_dim]
        # pos_items_emb: [batch_size, seq_len, embed_dim]
        logits = F.normalize(logits, dim=-1)
        pos_items_emb, _ = self.model.embedding(seqs)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)

        batch_size, seq_len, _ = logits.shape

        # Positive logits: dot product at same position
        pos_logits = (logits * pos_items_emb).sum(dim=-1) / 0.2  # [batch_size, seq_len]
        pos_logits = torch.exp(pos_logits)

        # Negative logits: compare each sequence with all others in batch at same position
        # [batch_size, seq_len, embed_dim] x [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, batch_size]
        neg_logits = torch.einsum('bld,mld->blm', logits, pos_items_emb) / 0.2  # [batch_size, seq_len, batch_size]

        # Mask out self-comparisons (diagonal in batch)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=seqs.device)  # [batch_size, batch_size]
        mask = mask.unsqueeze(1).expand(batch_size, seq_len, batch_size)     # [batch_size, seq_len, batch_size]
        neg_logits = torch.where(mask, neg_logits, torch.tensor(0.0, device=neg_logits.device))

        neg_logits = torch.exp(neg_logits).sum(dim=-1)  # [batch_size, seq_len]

        # Final loss
        loss = -torch.log(pos_logits / (neg_logits + 1e-8))  # avoid division by zero
        return loss.mean()

    def calculate_metrics(self, batch):
        seqs, labels = batch
        
        if self.args.dataset_code != 'xlong':
            scores = self.model(seqs)[0][:, -1, :]
            B, L = seqs.shape
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        else:
            scores, labels = self.model(seqs, labels=labels)
            scores = scores[:, -1, :]
        
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics