import os, math, gc, importlib, types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from RWKV.v6.state import BlockStateList
from typing import Union, Optional, List
from config import global_config


HEAD_SIZE = global_config.train_service_config.model.head_size

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop

CHUNK_LEN = global_config.train_service_config.model.chunk_len

full_parent_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'{full_parent_dir}/v7/cuda/wkv7_cuda.cu', f'{full_parent_dir}/v7/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)


class RWKV_Tmix_x070(MyModule):
    def __init__(self, layer_id, n_layer, n_embd, dim_att, head_size_divisor=8):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.dim_att = dim_att
        self.head_size_divisor = head_size_divisor

        self.head_size = HEAD_SIZE
        self.n_head = dim_att // self.head_size
        assert dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id!=0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    

class RWKV_CMix_x070(MyModule):
    def __init__(self, layer_id, n_embd, n_layer):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(n_embd**0.5), 0.5/(n_embd**0.5))
        # self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)

class Block(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd, dim_att, head_size_divisor=8, dropout=0):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_Tmix_x070(layer_id, n_layer, n_embd, dim_att, head_size_divisor)
        self.ffn = RWKV_CMix_x070(layer_id, n_embd, n_layer)

        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        if self.dropout > 0:
            x = self.drop0(x)

        x = x + self.ffn(self.ln2(x))
        
        if self.dropout > 0:
            x = self.drop1(x)
            
        return x, v_first


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

        
class HelperFunc:
    """
    框架辅助类
    """
    @staticmethod
    def calc_model_args(model: dict, head_size: int):
        if isinstance(model, str):
            model = torch.load(model, map_location='cpu')
        n_layer = 0
        for key in model.keys():
            if key.startswith("blocks."):
                n_layer += 1
        n_embd = model['head.weight'].shape[1]
        vocab_size = model['head.weight'].shape[0]
        dim_att = n_embd
        n_head = dim_att // head_size
        dim_ffn = int((n_embd * 3.5) // 32 * 32)
        return n_layer, n_embd, vocab_size, dim_att, n_head, dim_ffn



class RWKV(nn.Module):
    """
    模型框架类
    """
    def __init__(self, model, head_size: int , head_size_divisor: int = 8):
        super().__init__()
        self.dtype = torch.bfloat16
        if isinstance(model, str):
            model_weights = torch.load(model, map_location='cpu')
        elif isinstance(model, dict):
            model_weights = model
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
        
        _args = HelperFunc.calc_model_args(model, head_size)
        n_layer, n_embd, vocab_size, dim_att, n_head, dim_ffn = _args
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(i, n_layer, n_embd, dim_att, head_size_divisor) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # 加载model_weights
        self.apply_load_model(self, model_weights, self.dtype)
        self.params_v_first =  None
        self.params_states = None


    def get_v_first(self):
        if self.params_v_first is None:
            self.params_v_first = torch.empty_like(self.emb.weight)
        return self.params_v_first.cuda()
    
    def set_v_first(self, v_first):
        self.params_v_first = v_first.detach().cpu()
        return self.params_v_first

    def get_states(self, B, C, H, device, dtype):
        if self.params_states is None:
            self.params_states = BlockStateList.create(self.n_layer, B, C, H, device, dtype)
        return self.params_states.cuda()
    
    def set_states(self, states):
        self.params_states = states.detach().cpu()
        return self.params_states
    

    def action_load_model(self, model_weights:dict, dtype:torch.dtype):
        # 加载model_weights
        self.load_state_dict(model_weights)
        # 将model的参数转换为dtype
        for p in self.parameters():
            p.data = p.data.to(dtype=dtype)
        # 释放model_weights
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()
        # 返回model
        return self
    
    def action_add_dropout(self, dropout:float):
        self.drop0 = nn.Dropout(p=dropout)
        return self
    
    def get_optim_groups(self, 
                         weight_decay:float=0.0, 
                         layerwise_lr:float=0.0, 
                         my_pile_stage:int=1):
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_sta" in n) and (weight_decay > 0)):
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
            elif (len(p.squeeze().shape) >= 2) and (weight_decay > 0) and (".weight" in n):
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
                    {"params": [param_dict[n] for n in lr_1x], 
                     "weight_decay": 0.0, 
                     "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], 
                     "weight_decay": 0.0, 
                     "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], 
                     "weight_decay": 0.0, 
                     "my_lr_scale": 5.0},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], 
                     "weight_decay": 0.0, 
                     "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], 
                     "weight_decay": 0.0, 
                     "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], 
                             "weight_decay": 0.0, 
                             "my_lr_scale": 1.0}]

        if weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], 
                              "weight_decay": weight_decay, 
                              "my_lr_scale": 1.0}]

        return optim_groups
    
    def get_optimizer(self, 
                    optim_groups, 
                    lr_init:float=global_config.train_service_config.train.lr_init, 
                    beta1:float=global_config.train_service_config.train.beta1, 
                    beta2:float=global_config.train_service_config.train.beta2, 
                    eps:float=global_config.train_service_config.train.adam_eps, 
                    adamw_mode:bool=global_config.train_service_config.train.adamw_mode, 
                    weight_decay:float=global_config.train_service_config.train.weight_decay):
        
        optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=lr_init,
                betas=(beta1, beta2),
                eps=eps,
                adamw_mode=adamw_mode,
                weight_decay=weight_decay,
                amsgrad=False,
                bias_correction=True,
            )

        return optimizer
    
    def get_lr_scheduler(self, 
                        optimizer, 
                        warmup_min_lr:float=global_config.train_service_config.train.lr_final,
                        warmup_max_lr:float=global_config.train_service_config.train.lr_init, 
                        warmup_num_steps:int=global_config.train_service_config.train.warmup_steps, 
                        warmup_type:str=global_config.train_service_config.train.warmup_type):
        lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
            optimizer,
            warmup_min_lr=warmup_min_lr,
            warmup_max_lr=warmup_max_lr,
            warmup_num_steps=warmup_num_steps,
            warmup_type=warmup_type,
        )
        return lr_scheduler


    def forward(self, idx: Union[torch.Tensor, list], states: BlockStateList = None):
        # 计算logits
        args = self.args

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * args.head_size
        
        x = self.emb(idx)

        if args.dropout > 0:
            x = self.drop0(x)

        v_first = self.get_v_first()

        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        logits = self.head(x)

        self.set_v_first(v_first)
        # clean states
        return logits, None