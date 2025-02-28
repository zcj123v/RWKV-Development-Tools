import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any

class RWKVState:
    """
    RWKV模型的状态容器，用于RNN模式下的状态管理
    
    属性:
        batch_size: 批处理大小
        n_layer: 模型层数
        n_embd: 嵌入维度
        head_size: 注意力头大小
        n_head: 注意力头数量
        device: 状态所在设备
        dtype: 状态的数据类型
        v_first: 第一层的v状态
        layer_states: 每层的状态字典
    """
    def __init__(self, n_layer, n_embd, head_size, batch_size=1, device="cuda", dtype=torch.bfloat16):
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.head_size = head_size
        self.n_head = n_embd // head_size
        self.device = device
        self.dtype = dtype
        
        # 初始化为None，稍后会在init_state中创建实际张量
        self.v_first = None
        self.layer_states = [{} for _ in range(n_layer)]
    
    def init_state(self, model=None):
        """
        初始化RNN模式的状态
        
        参数:
            model: 可选的RWKV模型实例，用于从模型获取参数
            
        返回:
            self: 返回自身以支持链式调用
        """
        device = self.device
        B = self.batch_size
        C = self.n_embd
        H = self.n_head
        N = self.head_size
        L = self.n_layer
        
        # 确保嵌入维度是有效的正数
        if C <= 0 and model is not None:
            C = model.n_embd
            self.n_embd = C
        
        if C <= 0:
            raise ValueError(f"无效的嵌入维度: {C}，请提供有效的n_embd值或model参数")
        
        # 初始化v_first状态
        self.v_first = torch.zeros(B, 1, C, dtype=self.dtype, device=device)
        
        # 为每一层初始化状态
        for i in range(L):
            # 时间混合层状态
            self.layer_states[i]['att'] = {
                # x的前一个状态
                'x_prev': torch.zeros(B, C, dtype=self.dtype, device=device),
                # 注意力状态矩阵 (对应rnn.py中的state矩阵)
                'state': torch.zeros(B, H, N, N, dtype=torch.float32, device=device),
                # 时间衰减状态
                'w_state': torch.zeros(B, H, dtype=self.dtype, device=device),
                # 键值状态
                'k_state': torch.zeros(B, C, dtype=self.dtype, device=device),
                'v_state': torch.zeros(B, C, dtype=self.dtype, device=device),
                # 接收状态
                'r_state': torch.zeros(B, C, dtype=self.dtype, device=device),
                # 注意力激活状态
                'a_state': torch.zeros(B, C, dtype=self.dtype, device=device),
            }
            
            # 通道混合层状态
            self.layer_states[i]['ffn'] = {
                # x的前一个状态
                'x_prev': torch.zeros(B, C, dtype=self.dtype, device=device),
            }
        
        return self
    
    def reset_state(self):
        """
        重置所有状态为零
        
        返回:
            self: 返回自身以支持链式调用
        """
        # 重置v_first
        if self.v_first is not None:
            self.v_first.zero_()
        
        # 重置每层状态
        for i in range(self.n_layer):
            if self.layer_states[i]:
                for key, state_group in self.layer_states[i].items():
                    for state_name, state in state_group.items():
                        state.zero_()
        
        return self
    
    def to(self, device=None, dtype=None):
        """
        将状态移动到指定设备和/或转换为指定数据类型
        
        参数:
            device: 目标设备
            dtype: 目标数据类型
            
        返回:
            self: 返回自身以支持链式调用
        """
        if device is not None:
            self.device = device
        
        if dtype is not None:
            self.dtype = dtype
        
        # 移动v_first
        if self.v_first is not None:
            self.v_first = self.v_first.to(device=device, dtype=dtype)
        
        # 移动每层状态
        for i in range(self.n_layer):
            if self.layer_states[i]:
                for key, state_group in self.layer_states[i].items():
                    for state_name, state in state_group.items():
                        # 注意力状态矩阵保持float32精度
                        if state_name == 'state':
                            self.layer_states[i][key][state_name] = state.to(device=device)
                        else:
                            self.layer_states[i][key][state_name] = state.to(device=device, dtype=dtype)
        
        return self
    
    def detach(self):
        """
        分离状态的计算图，用于推理时避免内存泄漏
        
        返回:
            self: 返回自身以支持链式调用
        """
        # 分离v_first
        if self.v_first is not None:
            self.v_first = self.v_first.detach()
        
        # 分离每层状态
        for i in range(self.n_layer):
            if self.layer_states[i]:
                for key, state_group in self.layer_states[i].items():
                    for state_name, state in state_group.items():
                        self.layer_states[i][key][state_name] = state.detach()
        
        return self
    
    def clone(self):
        """
        创建状态的深拷贝
        
        返回:
            新的RWKVState实例
        """
        new_state = RWKVState(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            head_size=self.head_size,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype
        )
        
        # 复制v_first
        if self.v_first is not None:
            new_state.v_first = self.v_first.clone()
        
        # 复制每层状态
        for i in range(self.n_layer):
            if self.layer_states[i]:
                for key, state_group in self.layer_states[i].items():
                    if key not in new_state.layer_states[i]:
                        new_state.layer_states[i][key] = {}
                    for state_name, state in state_group.items():
                        new_state.layer_states[i][key][state_name] = state.clone()
        
        return new_state
    
    def cpu(self):
        """
        将状态移动到CPU
        
        返回:
            self: 返回自身以支持链式调用
        """
        return self.to(device="cpu")
    
    def cuda(self, device=None):
        """
        将状态移动到CUDA设备
        
        参数:
            device: 可选的CUDA设备ID
            
        返回:
            self: 返回自身以支持链式调用
        """
        if device is None:
            device = "cuda"
        elif isinstance(device, int):
            device = f"cuda:{device}"
        
        return self.to(device=device)
    
    def half(self):
        """
        将状态转换为半精度浮点数
        
        返回:
            self: 返回自身以支持链式调用
        """
        return self.to(dtype=torch.float16)
    
    def bfloat16(self):
        """
        将状态转换为bfloat16精度
        
        返回:
            self: 返回自身以支持链式调用
        """
        return self.to(dtype=torch.bfloat16)
    
    def float(self):
        """
        将状态转换为单精度浮点数
        
        返回:
            self: 返回自身以支持链式调用
        """
        return self.to(dtype=torch.float32)
    
    @classmethod
    def create_empty_state(cls, model):
        """
        根据模型创建空状态
        
        参数:
            model: RWKV模型实例
            
        返回:
            新的RWKVState实例
        """
        # 从模型获取参数
        n_layer = model.n_layer
        n_embd = model.n_embd
        head_size = model.blocks[0].att.head_size
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # 创建并初始化状态
        state = cls(
            n_layer=n_layer,
            n_embd=n_embd,
            head_size=head_size,
            device=device,
            dtype=dtype
        )
        
        return state.init_state(model)
    
