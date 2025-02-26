import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


class HelperFunc:
    """
    框架辅助类
    """
    @staticmethod
    def calc_model_args(model_weights: dict, head_size: int):
        n_layer = 0
        for key in model_weights.keys():
            if key.startswith("blocks."):
                n_layer += 1
        n_embd = model_weights['head.weight'].shape[1]
        vocab_size = model_weights['head.weight'].shape[0]
        dim_att = n_embd
        n_head = dim_att // head_size
        dim_ffn = int((n_embd * 3.5) // 32 * 32)
        return n_layer, n_embd, vocab_size, dim_att, n_head, dim_ffn


class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        


class ModelFramework(nn.Module):
    """
    模型框架类
    """
    def __init__(self, model, head_size: int):
        super().__init__()
        self.dtype = torch.bfloat16
        if isinstance(model, str):
            model_weights = torch.load(model, map_location='cpu')
        elif isinstance(model, dict):
            model_weights = model
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
        
        _args = HelperFunc.calc_model_args(model_weights, head_size)
        n_layer, n_embd, vocab_size, dim_att, n_head, dim_ffn = _args
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block( i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # 加载model_weights
        self.apply_load_model(self, model_weights, self.dtype)

    def apply_load_model(self, model_weights:dict, dtype:torch.dtype):
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
    
    def apply_add_dropout(self, dropout:float):
        self.drop0 = nn.Dropout(p=dropout)
        return self
    
    def get_optim_groups(self, weight_decay:float=0.0, layerwise_lr:float=0.0, my_pile_stage:int=1):
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

        if weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": weight_decay, "my_lr_scale": 1.0}]

        return optim_groups
    
    def get_optimizer(self, optim_groups, lr_init:float, beta1:float, beta2:float, eps:float, adamw_mode:bool, weight_decay:float, warmup_min_lr:float, warmup_max_lr:float, warmup_num_steps:int, warmup_type:str):
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


    def forward(self, x):
        ""
        for component in self.components:
            x = component(x)
        return x


