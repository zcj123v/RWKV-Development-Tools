import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import math

from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.state import RWKVState
from RWKV.v7.Dual.cuda import RUN_CUDA_RWKV7g
class RWKV_Tmix_Dual(nn.Module):
    """
    RWKV时间混合层的双模式实现
    支持Transformer并行模式和RNN序列模式
    """
    def __init__(self, dim_att, n_embd, n_layer, head_size_divisor, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = 64  # 默认头大小
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
            D_DECAY_LORA = max(32, int(round((1.8*(C**0.5))/32)*32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5)  # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8*(C**0.5))/32)*32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3*(C**0.5))/32)*32))  # suggestion
            if self.layer_id != 0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6*(C**0.8))/32)*32))  # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            # 层归一化，用于对注意力输出进行归一化处理
            # H: 注意力头数量，作为分组数
            # C: 嵌入维度，即特征通道数
            # eps: 添加到分母的小常数，防止除零错误
            #      根据head_size_divisor进行缩放，以适应不同的头大小
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(head_size_divisor**2))  # !!! notice eps value !!!
        
    def forward(self, x, v_first=None, state=None, mode=RWKVMode.TRANSFORMER):
        """
        统一的前向传播接口，根据模式选择不同的实现
        
        参数:
            x: 输入张量
            v_first: Transformer模式下的v_first状态
            state: RNN模式下的状态
            mode: 运行模式
            
        返回:
            输出张量和更新的状态
        """
        if mode == RWKVMode.TRANSFORMER:
            return self._forward_transformer(x, v_first)
        else:
            return self._forward_rnn(x, state)
    
    def _forward_transformer(self, x, v_first):
        """
        Transformer模式的前向传播实现
        
        参数:
            x: 输入张量 [B, T, C]
            v_first: 第一层的v状态
            
        返回:
            输出张量和更新的v_first
        """
        B, T, C = x.size()
        H = self.n_head
        
        # 时间混合
        xx = self.time_shift(x) - x

        # 应用时间混合系数
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # 计算各个组件
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        
        # 处理第一层的v状态
        if self.layer_id == 0:
            v_first = v  # 存储第一层的v
        else:
            # 添加值残差
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            
        # 计算注意力系数
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # a是"上下文学习率"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # 归一化k
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a-1) * self.k_a)

        # 使用CUDA核心计算
        x = self._run_rwkv7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 添加残差连接
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        
        return x, v_first
    
    def _forward_rnn(self, x, state):
        """
        RNN模式的前向传播实现
        
        参数:
            x: 输入张量 [B, 1, C]
            state: 当前层的状态
            
        返回:
            输出张量和更新的状态
        """
        B, T, C = x.size()  # B: 批次大小(batch size), T: 序列长度(sequence length, 在RNN模式下为1), C: 嵌入维度(embedding dimension)
        H = self.n_head
        N = self.head_size
        
        assert T == 1, "RNN模式只支持序列长度为1的输入"
        
        # 获取当前层的状态
        if self.layer_id >= len(state.layer_states):
            raise ValueError(f"状态中缺少层 {self.layer_id} 的信息")
        
        layer_state = state.layer_states[self.layer_id]['att']
        x_prev = layer_state['x_prev']  # 前一个x状态
        att_state = layer_state['state']  # 注意力状态矩阵 [B, H, N, N]
        
        # 从3D张量 [B, 1, C] 转换为2D张量 [B, C]
        x = x.squeeze(1)
        
        # 计算时间混合
        xx = x_prev - x
        
        # 应用时间混合系数
        xr = x + xx * self.x_r.squeeze(0).squeeze(0)
        xw = x + xx * self.x_w.squeeze(0).squeeze(0)
        xk = x + xx * self.x_k.squeeze(0).squeeze(0)
        xv = x + xx * self.x_v.squeeze(0).squeeze(0)
        xa = x + xx * self.x_a.squeeze(0).squeeze(0)
        xg = x + xx * self.x_g.squeeze(0).squeeze(0)
        
        # 计算各个组件
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0.squeeze(0).squeeze(0) + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        # 处理第一层的v状态
        if self.layer_id == 0:
            v_first = v  # 存储第一层的v
            state.v_first = v.unsqueeze(1)  # 更新状态中的v_first
        else:
            # 确保v_first存在
            if state.v_first is None:
                raise ValueError("状态中缺少v_first")
            
            v_first = state.v_first.squeeze(1)  # 从状态中获取v_first
            # 添加值残差
            v = v + (v_first - v) * torch.sigmoid(self.v0.squeeze(0).squeeze(0) + (xv @ self.v1) @ self.v2)
        
        # 计算注意力系数
        a = torch.sigmoid(self.a0.squeeze(0).squeeze(0) + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        # 归一化k
        kk = k * self.k_k.squeeze(0).squeeze(0)
        kk = F.normalize(kk.view(B, H, -1), dim=-1, p=2.0).view(B, C)
        k = k * (1 + (a-1) * self.k_a.squeeze(0).squeeze(0))
        
        # 将w转换为衰减因子
        w = torch.exp(-torch.exp(w))
        
        # 重塑张量以匹配状态形状
        r_reshaped = r.view(B, H, N)
        k_reshaped = k.view(B, H, N)
        v_reshaped = v.view(B, H, N)
        kk_reshaped = kk.view(B, H, N)
        a_reshaped = a.view(B, H, N)
        w_reshaped = w.view(B, H, N)
        
        # 初始化输出张量
        # B: 批次大小(batch size), H: 注意力头数量(number of heads), N: 每个头的大小(head size)
        out = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        
        # 对每个批次和头分别处理
        for b_idx in range(B):
            for h in range(H):
                # 获取当前状态
                s = att_state[b_idx, h]  # [N, N]
                
                # 获取当前输入
                wt = w_reshaped[b_idx, h].unsqueeze(0)  # [1, N]
                rt = r_reshaped[b_idx, h]  # [N]
                kt = k_reshaped[b_idx, h]  # [N]
                vt = v_reshaped[b_idx, h]  # [N]
                at = -kk_reshaped[b_idx, h]  # [N]
                bt = kk_reshaped[b_idx, h] * a_reshaped[b_idx, h]  # [N]
                
                # 计算外积
                vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)  # [N, N]
                ab = at.unsqueeze(-1) @ bt.unsqueeze(-2)  # [N, N]
                
                # 确保数据类型一致
                s_dtype = s.dtype
                ab = ab.to(dtype=s_dtype)
                vk = vk.to(dtype=s_dtype)
                
                # 更新状态 - 确保维度匹配
                # wt应该是标量或可广播到[N, N]的张量
                wt_broadcast = wt.unsqueeze(-1)  # [1, N, 1]
                s = s * wt_broadcast + s @ ab + vk
                
                # 保存更新后的状态
                att_state[b_idx, h] = s
                
                # 计算输出 - 确保rt与s兼容
                rt = rt.to(dtype=s_dtype)
                out[b_idx, h] = (s @ rt.unsqueeze(-1)).squeeze(-1).to(dtype=x.dtype)
        
        # 重塑输出
        out = out.reshape(B, C)
        
        # 应用层归一化 - 修改为与 rnn.py 一致的方式
        out = F.group_norm(out.view(B, H*N), num_groups=H, weight=self.ln_x.weight, bias=self.ln_x.bias, eps=self.ln_x.eps).view(B, C)
        
        # 添加残差连接
        out = out + ((r_reshaped * k_reshaped * self.r_k).sum(dim=-1, keepdim=True) * v_reshaped).view(B, C)
        
        # 应用输出变换
        out = self.output(out * g)
        
        # 更新状态
        layer_state['x_prev'] = x.clone()  # 更新x状态
        
        # 返回输出和更新的状态
        return out.unsqueeze(1), state  # 添加T维度
    
    def _run_rwkv7g(self, q, w, k, v, a, b):
        """
        RWKV v7 的CUDA核心计算函数包装器
        
        参数:
            q, w, k, v, a, b: 输入张量
            
        返回:
            输出张量
        """
        B, T, HC = q.shape
        
        # 尝试使用CUDA核心实现
        try:

            return RUN_CUDA_RWKV7g(q, w, k, v, a, b)
        except (ImportError, RuntimeError) as e:
            # 如果CUDA核心不可用或出错，使用PyTorch实现
            import logging
            logging.warning(f"CUDA核心不可用或出错: {e}，使用PyTorch实现（性能较低）")
            print(f"CUDA核心不可用或出错: {e}，使用PyTorch实现（性能较低）")
            # 确保所有输入都是相同的数据类型
            dtype = q.dtype
            H = HC // self.head_size
            N = self.head_size
            
            # 重塑为更易于处理的形状
            q = q.view(B, T, H, N)
            w = w.view(B, T, H, N)
            k = k.view(B, T, H, N)
            v = v.view(B, T, H, N)
            a = a.view(B, T, H, N)
            b = b.view(B, T, H, N)
            
            # 初始化输出
            y = torch.zeros_like(v)
            
            # 将w转换为衰减因子
            w = torch.exp(-torch.exp(w))
            
            # 对每个批次和头分别处理
            for b_idx in range(B):
                for h in range(H):
                    # 初始化状态矩阵
                    s = torch.zeros(N, N, device=q.device, dtype=torch.float32)
                    
                    # 按时间步处理
                    for t in range(T):
                        # 获取当前时间步的张量
                        wt = w[b_idx, t, h].unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
                        qt = q[b_idx, t, h]  # [N]
                        kt = k[b_idx, t, h]  # [N]
                        vt = v[b_idx, t, h]  # [N]
                        at = a[b_idx, t, h]  # [N]
                        bt = b[b_idx, t, h]  # [N]
                        
                        # 计算外积
                        vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)  # [N, N]
                        ab = at.unsqueeze(-1) @ bt.unsqueeze(-2)  # [N, N]
                        
                        # 确保数据类型一致 - 将ab转换为float32以匹配s
                        ab = ab.to(dtype=torch.float32)
                        vk = vk.to(dtype=torch.float32)
                        wt = wt.to(dtype=torch.float32)
                        
                        # 更新状态
                        s = s * wt + s @ ab + vk
                        
                        # 计算输出 - 确保qt与s兼容
                        qt_float = qt.to(dtype=torch.float32)
                        y[b_idx, t, h] = (s @ qt_float.unsqueeze(-1)).squeeze(-1).to(dtype=dtype)
            
            # 重塑回原始形状
            return y.reshape(B, T, HC)

