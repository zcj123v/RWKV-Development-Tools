import torch
from torch.nn import functional as F
from .model import RWKV, L2Wrap
import numpy as np
import copy
import sys

def train_forward(rwkv, batch: dict, v_first=None):
    # pre calc
    seq = batch["input_ids"]
    mask = batch.get("masks", None)
    # data process
    idx = seq[:-1]
    targets = seq[1:]
    if mask == None:
        mask = [int(x != 0) for x in idx]
    else:
        mask = mask[:-1]
    # data into tensor
    targets = torch.tensor([targets], dtype=torch.long).cuda()

    # process mask
    mask = torch.tensor([mask], dtype=torch.float32).cuda()
    # .to(next(rwkv.parameters()).device)
    mask = mask.view(-1)
    logits, v_first = rwkv(idx, v_first)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
    )

    sum_mask = mask.sum().clamp(min=1e-8)  # 防止除零的守护符
    masked_loss = loss * mask
    final_loss = torch.where(
        sum_mask > 0,
        masked_loss.sum() / sum_mask,
        masked_loss.sum()
    )
    

    return L2Wrap.apply(final_loss, logits), v_first

