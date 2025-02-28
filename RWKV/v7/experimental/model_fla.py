import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import types
from RWKV.v7.experimental.state import BlockStateList,BlockState, TimeMixState, ChannelMixState
from typing import Union, Optional, List
from fla.ops.rwkv7 import chunk_rwkv7

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
# if os.environ["RWKV_JIT_ON"] == "1":
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method

def RUN_RWKV7_INFCTX(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    w=-torch.exp(w)
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    o, state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
    return o, state


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
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
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()


    def forward(self, x, v_first, last_state: TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        #xx = self.time_shift(x) - x
        
        shift_state = last_state.shift_state
        wkv_state = last_state.wkv_state.clone().contiguous() 

        xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x


        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        #print(f'x shape = {x.shape}')

        shift_state = x[:,-1,:]

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

        x , wkv_state = RUN_RWKV7_INFCTX(r,k,v,w,-kk, kk*a,wkv_state)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        
        return x, v_first,TimeMixState(shift_state,wkv_state)


class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

    def forward(self, x,last_state: ChannelMixState):
        #xx = self.time_shift(x) - x
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :0]), dim=1) - x
        
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k), ChannelMixState(x[:, -1])
    

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)


        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
    
    def forward(self, x, v_first, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state)
        x = x + x_attn

        ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)

        x = x + ffn_out
        return x, v_first, BlockState(att_state, ffn_state)

class L2Wrap(torch.autograd.Function):
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
        if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
            # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)


class RWKV(nn.Module):
    def __init__(self, args_in):
        super().__init__()
        args = types.SimpleNamespace()
        args.n_embd = args_in.model.n_embd
        args.n_layer = args_in.model.n_layer
        args.vocab_size = args_in.model.vocab_size
        args.dropout = args_in.train.dropout
        args.grad_cp = 1
        args.lora_on = args_in.lora.lora_on
        args.chunk_len  = args_in.model.chunk_len
        args.ctx_len = args_in.model.ctx_len
        args.head_size = args_in.model.head_size
        args.head_size_divisor = args_in.model.head_size_divisor
        args.load_model = args_in.model.load_model
        args.lora = args_in.lora
        args.trainer = args_in.train
        args.model = args_in.model
        args.weight_decay = args_in.train.weight_decay
        self.args = args

        # 统一dtype处理
        dtype_map = {
            "fp32": torch.float,
            "fp16": torch.half,
            "bf16": torch.bfloat16
        }
        self.dtype = dtype_map.get(args_in.model.dtype, torch.bfloat16)
        
        if self.args.model.dtype == "fp32":
            self.args.model.dtype = torch.float
        elif self.args.model.dtype == "fp16":
            self.args.model.dtype = torch.half
        elif self.args.model.dtype == "bf16":
            self.args.model.dtype = torch.bfloat16

        # load weight
        model_weights = torch.load(args.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())

        # calc init layer
        if args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            args.n_layer = max_block_id + 1

        # calc n_embd
        if args.n_embd < 0:
            args.n_embd = model_weights['head.weight'].shape[1]

        # clac vocab_size
        if args.vocab_size < 0:
            args.vocab_size = model_weights['head.weight'].shape[0]

        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)


        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # init dropout
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

        model_weights = {k:v for k,v in model_weights.items()}
        self.load_state_dict(model_weights)

        for p in self.parameters():
            p.data = p.data.to(dtype=self.args.model.dtype)

        del model_weights
        gc.collect()
        torch.cuda.empty_cache()

    def get_optim_groups(self):
        args = self.args.trainer
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_sta" in n) and (args.weight_decay > 0)):
                lr_decay.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

        optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=self.args.trainer.lr_init,
                betas=(self.args.trainer.beta1, self.args.trainer.beta2),
                eps=self.args.trainer.adam_eps,
                adamw_mode=self.args.trainer.adamw_mode,
                weight_decay=self.args.trainer.weight_decay,
                amsgrad=False,
                bias_correction=True,
            )

        lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
            optimizer,
            warmup_min_lr=self.args.trainer.lr_final,
            warmup_max_lr=self.args.trainer.lr_init,
            warmup_num_steps=self.args.trainer.warmup_steps,
            warmup_type="linear",
        )

        return optimizer, lr_scheduler


    def inference_forward(self, idx, states: BlockStateList = None):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H =  args.dim_att // args.head_size
        assert C==H*args.head_size
        
        if states is None:
            states = BlockStateList.create(args.n_layer, B, C, H, idx.device,self.emb.weight.dtype)
        
        states = states.to_cuda()
        last_shift_states = states.shift_states
        last_wkv_states = states.wkv_states

        # 初始化new_states，作为输出
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H, x.device, x.dtype)
        
        # 初始化v_first，作为残差
        if states.v_first is None:
            v_first = torch.empty_like(x)
        else:
            v_first = states.v_first.to(x.device, x.dtype)

        # 初始化x,词嵌入
        x = self.emb(idx)
        
        # 遍历blocks
        for i, (block, block_state) in enumerate(zip(self.blocks, BlockStateList(last_shift_states, last_wkv_states))):
            if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, use_reentrant=False)
            else:
                x, v_first, new_block_state = block(x,v_first,block_state)

            new_states[i] = new_block_state 

        # 输出层
        x = self.ln_out(x)    
        x = self.head(x)

        # 将new_states转换为cpu
        new_states =  new_block_state.to_cpu()
        new_states.set_vfirst(v_first.detach().clone())

        return x, new_states

    def forward_raw(self, idx, v_first, last_shift_states: torch.Tensor, last_wkv_states: torch.Tensor):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H =  args.dim_att // args.head_size
        assert C==H*args.head_size
        
        x = self.emb(idx)
        # 检查并处理embedding输出的nan/inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                        x.device, x.dtype)
        if v_first is None:
            v_first = torch.empty_like(x)
        else:
            v_first = v_first.to(x.device, x.dtype)
            v_first = torch.nan_to_num(v_first, nan=0.0, posinf=1e6, neginf=-1e6)
        
        for i, (block, block_state) in enumerate(zip(self.blocks, BlockStateList(last_shift_states, last_wkv_states))):
            if args.grad_cp == 1 and i > 0:# and i < len(self.blocks)-1 :
                x, v_first, new_block_state = torch_checkpoint(block, x, v_first, block_state, use_reentrant=False)
            else:
                x, v_first, new_block_state = block(x,v_first,block_state)

            # 每个block后检查并处理nan/inf
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            v_first = torch.nan_to_num(v_first, nan=0.0, posinf=1e6, neginf=-1e6)
            new_states[i] = new_block_state 

        # 输出层前的LayerNorm
        x = self.ln_out(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 最终输出层
        x = self.head(x)
        
        # 最终输出的处理
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # 对输出进行clip,避免过大或过小的值
        x = torch.clamp(x, -100, 100)

        return x, v_first, new_states.shift_states, new_states.wkv_states

    def forward(self, idx, targets, states: BlockStateList = None):
        args = self.args
        T_train = args.chunk_len
        # idx, targets = batch
        B, T = idx.shape
        C = args.n_embd   
        H =  args.dim_att // args.head_size
        assert C == H*args.head_size

        # 初始化states
        if states is None:
            states = BlockStateList.create(args.n_layer, B, C, H, idx.device,self.emb.weight.dtype)

        # 初始化v_first
        if states.get_vfirst() is None:
            v_first = torch.empty_like(idx)
        else:
            v_first = states.get_vfirst()
        states.to_cuda()
        # state初始化结束

        # 定义checkpointed_step
        def checkpointed_step(idx, targets, prev_loss, v_first, last_shift_states, last_wkv_states, prev_token_amount):
            # print(f"idx.shape = {idx.shape}")
            logits, v_first, new_shift_states, new_wkv_states = self.forward_raw(idx, v_first, last_shift_states, last_wkv_states)
            
            # 检查logits是否包含nan或inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN or Inf detected in logits")
                # 将nan和inf替换为0
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            
            current_token_amount = (targets!=-100).sum() #这样是不是更合适？
            current_token_amount = idx.shape[1]
            
            # 添加数值稳定性的clip
            logits = torch.clamp(logits, -100, 100)
            
            if current_token_amount == 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction='sum')
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                loss = L2Wrap.apply(loss, logits, current_token_amount)
            
            # 检查loss是否为nan
            if torch.isnan(loss):
                print("Warning: NaN detected in loss")
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            
            new_token_amount = prev_token_amount + current_token_amount
            if new_token_amount > 0:
                # 添加eps避免除0
                eps = 1e-8
                new_loss = prev_loss * (prev_token_amount / (new_token_amount + eps)) + loss * (
                    current_token_amount / (new_token_amount + eps))
            else:
                new_loss = prev_loss
                
            # 最后再检查一次输出
            if torch.isnan(new_loss):
                print("Warning: NaN detected in final loss")
                new_loss = torch.tensor(0.0, device=new_loss.device, dtype=new_loss.dtype)
                
            return new_loss, v_first, new_shift_states, new_wkv_states, new_token_amount
        
        total_loss = torch.tensor(0.,dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        i = 0
        for i in range(math.ceil(T / T_train)):
            total_loss,v_first,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_train:(i + 1) * T_train],
                targets[:, i * T_train:(i + 1) * T_train],
                total_loss,
                v_first,
                states.shift_states,
                states.wkv_states,
                token_amount,
                use_reentrant=False
            )
            states = BlockStateList(new_shift_states.clone().detach(), new_wkv_states.clone().detach())
            print("total_loss====>", total_loss, token_amount)

        states.set_vfirst(v_first.detach().clone())
        return total_loss, states.to_cpu()



