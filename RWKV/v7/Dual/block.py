import torch
import torch.nn as nn
import torch.nn.functional as F

from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.time_mix import RWKV_Tmix_Dual
from RWKV.v7.Dual.channel_mix import RWKV_CMix_Dual
from RWKV.v7.Dual.state import RWKVState

class Block_Dual(nn.Module):
    """
    RWKV模型的基本块的双模式实现
    包含时间混合和通道混合层
    """
    def __init__(self, n_embd, dim_att, n_layer, head_size_divisor, dropout, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        
        # 使用双模式的时间混合和通道混合层
        self.att = RWKV_Tmix_Dual(dim_att, n_embd, n_layer, head_size_divisor, layer_id)
        self.ffn = RWKV_CMix_Dual(n_embd, n_layer, layer_id)
        
        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
    
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
            x: 输入张量
            v_first: 第一层的v状态
            
        返回:
            输出张量和更新的v_first
        """
        if self.layer_id == 0:
            x = self.ln0(x)
        
        # 时间混合层
        x_ln1 = self.ln1(x)
        x_attn, v_first = self.att(x_ln1, v_first=v_first, mode=RWKVMode.TRANSFORMER)
        
        # 应用dropout（如果启用）
        if self.dropout > 0:
            x_attn = self.drop0(x_attn)
            
        x = x + x_attn
        
        # 通道混合层
        x_ln2 = self.ln2(x)
        x_ffn, _ = self.ffn(x_ln2, mode=RWKVMode.TRANSFORMER)
        
        # 应用dropout（如果启用）
        if self.dropout > 0:
            x_ffn = self.drop1(x_ffn)
            
        x = x + x_ffn
        return x, v_first
    
    def _forward_rnn(self, x, state):
        """
        RNN模式的前向传播实现
        
        参数:
            x: 输入张量 [B, 1, C]
            state: RNN状态
            
        返回:
            输出张量和更新的状态
        """
        B, T, C = x.size()
        assert T == 1, "RNN模式只支持序列长度为1的输入"
        
        # 确保状态已正确初始化
        if state is None:
            raise ValueError("RNN模式需要提供状态")
            
        # 确保当前层的状态存在
        # print(f">>>> self.layer_id: {self.layer_id}, state.layer_states: {len(state.layer_states)}")
        if self.layer_id + 1 > len(state.layer_states):
            raise ValueError(f"状态中缺少层 {self.layer_id} 的信息")
        
        # 第0层需要额外的层归一化
        if self.layer_id == 0:
            x = self.ln0(x)
        
        # 时间混合层
        x_ln1 = self.ln1(x)
        x_attn, state = self.att(x_ln1, state=state, mode=RWKVMode.RNN)
        
        # 应用dropout（如果启用且在训练模式）
        if self.dropout > 0 and self.training:
            x_attn = self.drop0(x_attn)
            
        x = x + x_attn
        
        # 通道混合层
        x_ln2 = self.ln2(x)
        x_ffn, state = self.ffn(x_ln2, state=state, mode=RWKVMode.RNN)
        
        # 应用dropout（如果启用且在训练模式）
        if self.dropout > 0 and self.training:
            x_ffn = self.drop1(x_ffn)
            
        x = x + x_ffn
        
        return x, state
