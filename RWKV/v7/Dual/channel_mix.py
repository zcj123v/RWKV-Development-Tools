import torch
import torch.nn as nn
import torch.nn.functional as F

from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.state import RWKVState

class RWKV_CMix_Dual(nn.Module):
    """
    RWKV通道混合层的双模式实现
    支持Transformer并行模式和RNN序列模式
    """
    def __init__(self, n_embd, n_layer, layer_id):
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
        
    def forward(self, x, state=None, mode=RWKVMode.TRANSFORMER):
        """
        统一的前向传播接口，根据模式选择不同的实现
        
        参数:
            x: 输入张量
            state: RNN模式下的状态
            mode: 运行模式
            
        返回:
            输出张量和更新的状态
        """
        if mode == RWKVMode.TRANSFORMER:
            return self._forward_transformer(x), state
        else:
            return self._forward_rnn(x, state)
    
    def _forward_transformer(self, x):
        """
        Transformer模式的前向传播实现
        
        参数:
            x: 输入张量 [B, T, C]
            
        返回:
            输出张量
        """
        # 时间偏移
        xx = self.time_shift(x) - x
        
        # 应用通道混合
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
        
    def _forward_rnn(self, x, state):
        """
        RNN模式的前向传播实现
        
        参数:
            x: 输入张量 [B, 1, C]
            state: 当前层的状态
            
        返回:
            输出张量和更新的状态
        """
        B, T, C = x.size()
        assert T == 1, "RNN模式只支持序列长度为1的输入"
        
        # 获取当前层的状态
        if self.layer_id >= len(state.layer_states):
            raise ValueError(f"状态中缺少层 {self.layer_id} 的信息")
            
        layer_state = state.layer_states[self.layer_id]['ffn']
        x_prev = layer_state['x_prev']
        
        # 从3D张量 [B, 1, C] 转换为2D张量 [B, C]
        x = x.squeeze(1)
        
        # 实现RNN模式的前向传播逻辑，参考rnn.py中的channel_mixing函数
        xx = x_prev - x
        k = x + xx * self.x_k.squeeze(0).squeeze(0)
        
        # 确保使用与rnn.py相同的计算方式
        k = torch.relu(self.key(k)) ** 2
        
        # 计算输出
        output = self.value(k)
        
        # 更新状态
        layer_state['x_prev'] = x.clone()
        
        # 将输出转回3D张量 [B, 1, C]
        output = output.unsqueeze(1)
        
        return output, state
