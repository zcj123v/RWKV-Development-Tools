import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True) - 这些注释掉的代码是用于启用PyTorch的JIT性能分析功能
# torch._C._jit_set_profiling_mode(True) - 如果需要性能分析，可以取消注释
import torch.nn as nn
from torch.nn import functional as F
# 导入了基本的PyTorch模块和功能，代码结构清晰
# 建议：考虑是否需要所有这些导入，例如gc可能只在特定情况下使用
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import types
from RWKV.v7.triton.state import BlockState
from typing import Union, Optional, List
from RWKV.v7.triton.time_mix import RWKV_Tmix_x070
from RWKV.v7.triton.channel_mix import RWKV_CMix_x070
from RWKV.v7.triton.triton_kernels import RUN_CUDA_RWKV7g, RUN_RWKV7_STATE
from RWKV.v7.triton.state_manager import RWKVStateManager
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from RWKV.v7.triton.block import Block
from RWKV.v7.triton.state import BlockStateList
import torch.nn as nn

HEAD_SIZE = 64

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
criterion = nn.CrossEntropyLoss()


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    
class L2WrapInfctx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        return loss

    @staticmethod
    def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        if ctx.token_amount == 0:
            return (grad_output, None, None)
        factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        if True:
            maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)



class RWKV(nn.Module):
    def __init__(
        self, 
        # Model structure parameters
        n_embd=-1, 
        n_layer=-1, 
        vocab_size=-1,
        ctx_len=1024,
        head_size=64,
        head_size_divisor=8,
        # Training parameters
        dropout=0.0,
        grad_cp=1,
        # Model weight loading
        load_model="",
        # Data types
        dtype="bf16",
        # LoRA parameters
        lora_on=False,
        lora=None,
        # Training hyperparameters
        weight_decay=0.0,
        trainer=None
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor
        self.dropout = dropout
        self.grad_cp = grad_cp
        self.load_model = load_model
        self.lora_on = lora_on
        self.lora = lora
        self.weight_decay = weight_decay
        self.trainer = trainer
        self.param_vfirst = None

        # Unified dtype handling
        dtype_map = {
            "fp32": torch.float,
            "fp16": torch.half,
            "bf16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Load weights
        model_weights = torch.load(load_model, map_location='cpu')
        model_keys = list(model_weights.keys())

        # Calculate n_layer
        if self.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            self.n_layer = max_block_id + 1

        # Calculate n_embd
        if self.n_embd < 0:
            self.n_embd = model_weights['head.weight'].shape[1]

        # Calculate vocab_size
        if self.vocab_size < 0:
            self.vocab_size = model_weights['head.weight'].shape[0]

        self.dim_att = self.n_embd
        self.n_head = self.dim_att // self.head_size
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.blocks = nn.ModuleList([Block(n_embd=self.n_embd, 
                                           n_layer=self.n_layer, 
                                           layer_id=i, 
                                           dim_att=self.dim_att, 
                                           head_size=self.head_size, 
                                           head_size_divisor=self.head_size_divisor, 
                                           dropout=self.dropout) 
                                    for i in range(self.n_layer)])
        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Initialize dropout
        if self.dropout > 0:
            self.drop0 = nn.Dropout(p=self.dropout)

        model_weights = {k:v for k,v in model_weights.items()}
        self.load_state_dict(model_weights)

        for p in self.parameters():
            p.data = p.data.to(dtype=self.dtype)

        del model_weights
        gc.collect()
        torch.cuda.empty_cache()

    def get_optim_groups(self):
        if not self.trainer:
            raise ValueError("Trainer configuration is required for optimization groups")
            
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        
        my_pile_stage = self.trainer.get('my_pile_stage', 1)
        layerwise_lr = self.trainer.get('layerwise_lr', 0)
        
        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_sta" in n) and (self.weight_decay > 0)):
                lr_decay.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (self.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}
        
        if layerwise_lr > 0:
            if my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if self.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": self.weight_decay, "my_lr_scale": 1.0}]

        optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=self.trainer.get('lr_init', 1e-4),
                betas=(self.trainer.get('beta1', 0.9), self.trainer.get('beta2', 0.99)),
                eps=self.trainer.get('adam_eps', 1e-8),
                adamw_mode=self.trainer.get('adamw_mode', True),
                weight_decay=self.weight_decay,
                amsgrad=False,
                bias_correction=True,
            )

        lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
            optimizer,
            warmup_min_lr=self.trainer.get('lr_final', 1e-5),
            warmup_max_lr=self.trainer.get('lr_init', 1e-4),
            warmup_num_steps=self.trainer.get('warmup_steps', 1000),
            warmup_type="linear",
        )

        return optimizer, lr_scheduler

    def forward_without_state(self, idx: Union[torch.Tensor, list], v_first=None, state=None):
        # Convert input to tensor if needed
        if isinstance(idx, list):
            x = torch.tensor(idx, device=next(self.parameters()).device, dtype=torch.long)
        else:
            x = idx.to(device=next(self.parameters()).device, dtype=torch.long)

        B, T = x.size()
        C = self.n_embd
        H = self.dim_att // self.head_size

        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * self.head_size
        
        x = self.emb(x)

        if self.dropout > 0:
            x = self.drop0(x)

        if v_first is None:
            v_first = torch.empty_like(x)
        v_first = v_first.to("cuda")

        for block in self.blocks:
            if self.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, None

    def forward_with_state(self, idx, states=None):
        """Forward pass with explicit state management
        
        Args:
            idx: Input tensor of token indices [B, T]
            states: Optional list of layer states
            
        Returns:
            logits: Output logits
            new_states: Updated states for all layers
        """
        # Convert input to tensor if needed
        if isinstance(idx, list):
            x = torch.tensor(idx, device=next(self.parameters()).device, dtype=torch.long)
        else:
            x = idx.to(device=next(self.parameters()).device, dtype=torch.long)
        
        B, T = x.size()
        C = self.n_embd
        H = self.dim_att // self.head_size
        
        # Initialize new states
        new_states = [None] * self.n_layer
        
        # Initial embeddings
        x = self.emb(x)
        
        if self.dropout > 0:
            x = self.drop0(x)
        
        # Initialize v_first (value from first layer needed by later layers)
        v_first = None
        if states is None:
            states = BlockStateList.create(self.n_layer, B, C, H, idx.device, self.emb.weight.dtype)

        states = states.to_cuda()

        new_states = BlockStateList.create(self.n_layer, B, C, H, idx.device, self.emb.weight.dtype)

        for i , block in enumerate(self.blocks):
            state = states[i]
            if self.grad_cp == 1 and i > 0:
                x, v_first, new_time_state, new_channel_state = torch_checkpoint(block, x, v_first, state.time_mix_state, state.channel_mix_state, use_reentrant=False)
            else:
                x, v_first, new_time_state, new_channel_state = block(x, v_first, state.time_mix_state, state.channel_mix_state)
            new_states[i] = BlockState(new_time_state, new_channel_state)

        # Final layer norm and output
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, new_states.to_cpu()

    def forward(self, idx, states=None):
        """
        Forward pass that can work with or without state management
        
        Args:
            idx: Input tensor of token indices [B, T]
            states: Optional list of layer states
            
        Returns:
            If states is None:
                logits: Output logits
                None
            Otherwise:
                logits: Output logits
                new_states: Updated states for all layers
        """
        if states is None:
            return self.forward_without_state(idx)
        else:
            return self.forward_with_state(idx, states) 

    def training_step(self, batch, batch_idx):
        args = self.args
        T_train = args.chunk_ctx 
        idx, targets = batch
        B, T = idx.shape
        C = args.n_embd
        H =  args.dim_att // args.head_size_a
        assert C==H*args.head_size_a
        states = BlockStateList.create(args.n_layer, B, C, H, idx.device, self.emb.weight.dtype)

        def checkpointed_step(idx, targets, prev_loss, last_shift_states,
                            last_wkv_states, prev_token_amount):
            state = BlockStateList(last_shift_states, last_wkv_states)
            logits, state = self(idx, state)
            new_shift_states, new_wkv_states = state.shift_states, state.wkv_states
            current_token_amount = (targets!=-100).sum() #这样是不是更合适？
            current_token_amount = idx.shape[1]
            if current_token_amount == 0:
                loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1),reduction='sum')
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                loss = L2WrapInfctx.apply(loss, logits, current_token_amount)
            new_token_amount = prev_token_amount+current_token_amount
            if new_token_amount>0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                    current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss

            return new_loss, new_shift_states, new_wkv_states, new_token_amount
        
        total_loss = torch.tensor(0.,dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        i = 0
        for i in range(math.ceil(T / T_train)):

            total_loss,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_train:(i + 1) * T_train],
                targets[:, i * T_train:(i + 1) * T_train],
                total_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                use_reentrant=False
            )

        states = BlockStateList(new_shift_states.clone().detach(), new_wkv_states.clone().detach())
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return total_loss