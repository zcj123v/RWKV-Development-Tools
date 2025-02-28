"""
RWKV Dual Mode Architecture
---------------------------
This module implements a dual-mode architecture for RWKV models,
supporting both Transformer-style parallel training and RNN-style
sequential inference with the same model parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
from typing import Union, Optional, List, Tuple, Dict, Any

from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.block import Block_Dual
from RWKV.v7.Dual.state import RWKVState

# 配置日志记录器
logger = logging.getLogger(__name__)

class RWKV_Dual(nn.Module):
    """
    RWKV模型的双模式实现
    支持Transformer并行训练模式和RNN序列推理模式
    """
    def __init__(
        self,
        # 模型架构参数
        n_embd=-1,
        n_layer=-1,
        vocab_size=-1,
        head_size=64,
        head_size_divisor=8,
        ctx_len=1024,
        
        # 模型加载参数
        load_model=None,
        dtype="bf16",
        
        # 训练参数
        dropout=0.0,
        grad_cp=1,
        
        # 运行模式
        mode=RWKVMode.TRANSFORMER
    ):
        super().__init__()
        
        # 保存配置
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.head_size = head_size
        self.ctx_len = ctx_len
        self.mode = mode
        self.dropout = dropout
        self.grad_cp = grad_cp
        
        # 统一数据类型处理
        dtype_map = {
            "fp32": torch.float,
            "fp16": torch.half,
            "bf16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # 确保load_model已指定
        assert load_model is not None, "load_model必须指定"
        
        # 加载权重
        model_weights = torch.load(load_model, map_location='cpu')
        
        # 如果未指定，计算n_layer
        if n_layer < 0:
            max_block_id = max((int(x.split('.')[1]) for x in model_weights.keys() if 'blocks.' in x), default=0)
            n_layer = max_block_id + 1
        self.n_layer = n_layer
        # 如果未指定，计算n_embd
        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]
        self.n_embd = n_embd
        # 如果未指定，计算vocab_size
        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]
        self.vocab_size = vocab_size
        # 计算注意力维度
        dim_att = n_embd
        
        # 初始化模型组件
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        # 使用双模式块
        self.blocks = nn.ModuleList([
            Block_Dual(n_embd, dim_att, n_layer, head_size_divisor, dropout, i)
            for i in range(n_layer)
        ])
        
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 初始化dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
        
        # 加载状态字典
        self.load_state_dict(model_weights)
        
        # 将参数转换为指定的数据类型
        for p in self.parameters():
            p.data = p.data.to(dtype=self.dtype)
        
        # 清理
        del model_weights
        self.clear_gpu_memory(force=True)
    
    def set_mode(self, mode):
        """
        设置模型运行模式
        
        参数:
            mode: RWKVMode.TRANSFORMER 或 RWKVMode.RNN
            
        返回:
            self: 返回自身以支持链式调用
        """
        self.mode = mode
        return self
    
    def get_state_class(self):
        """
        获取适用于当前模型的状态类
        
        返回:
            RWKVState类
        """
        return RWKVState
    
    def create_state(self, batch_size=1, device=None):
        """
        创建并初始化RNN模式的状态
        
        参数:
            batch_size: 批处理大小
            device: 状态所在设备，默认使用模型设备
            
        返回:
            初始化的RWKVState实例
        """
        if device is None:
            device = self.get_device()
            
        state = RWKVState(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            head_size=self.head_size,
            batch_size=batch_size,
            device=device
        )
        
        return state.init_state(self)
    
    def forward(self, 
                idx: Union[torch.Tensor, list], 
                state: Optional[RWKVState] = None,
                v_first: Optional[torch.Tensor] = None,
                return_state: bool = True,
                return_v_first: bool = False):
        """
        统一的前向传播接口，根据模式选择不同的实现
        
        参数:
            idx: 输入张量或token ID列表
            state: RNN模式下的状态
            v_first: Transformer模式下的v_first状态
            return_state: 是否返回状态
            return_v_first: 是否返回v_first状态
            
        返回:
            根据模式和返回选项，返回不同的结果组合
        """
        if self.mode == RWKVMode.TRANSFORMER:
            return self._forward_transformer(idx, v_first, return_v_first)
        else:
            return self._forward_rnn(idx, state, return_state)
    
    def _forward_transformer(self, idx, v_first=None, return_v_first=False):
        """
        Transformer模式的前向传播实现
        
        参数:
            idx: 输入张量或token ID列表
            v_first: 第一层的v状态
            return_v_first: 是否返回v_first状态
            
        返回:
            logits: 输出logits
            v_first: (可选) 更新的v_first状态
        """
        # 转换输入为张量
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, device=self.get_device(), dtype=torch.long)
        else:
            idx = idx.to(self.get_device())
        
        B, T = idx.size()
        C = self.n_embd
        
        assert T <= self.ctx_len, f"无法处理长度为{T}的序列，模型ctx_len为{self.ctx_len}"
        
        # 嵌入输入
        x = self.emb(idx)
        x = self.optimize_tensor(x)
        
        if self.dropout > 0:
            x = self.drop0(x)
        
        # 初始化v_first
        if v_first is None:
            v_first = torch.zeros_like(x, device=x.device)
        else:
            v_first = self.optimize_tensor(v_first)
        
        # 通过各层处理
        for i, block in enumerate(self.blocks):
            if self.grad_cp == 1 and self.training:
                # 使用梯度检查点减少内存使用
                import deepspeed
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first, mode=RWKVMode.TRANSFORMER)
        
        # 输出层
        x = self.ln_out(x)
        logits = self.head(x)
        
        if return_v_first:
            return logits, v_first
        else:
            return logits
    
    def _forward_rnn(self, idx, state=None, return_state=True):
        """
        RNN模式的前向传播实现
        
        参数:
            idx: 输入张量或单个token ID
            state: RNN状态
            return_state: 是否返回更新的状态
            
        返回:
            logits: 输出logits
            state: (可选) 更新的状态
        """
        # 确保有状态
        if state is None:
            state = self.create_state()
        
        # 转换输入为张量
        if not isinstance(idx, torch.Tensor):
            if isinstance(idx, int):
                idx = torch.tensor([[idx]], device=self.get_device(), dtype=torch.long)
            else:
                idx = torch.tensor([idx], device=self.get_device(), dtype=torch.long)
        else:
            idx = idx.to(self.get_device())
            # 确保形状正确 [B, 1]
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)  # [1, T]
            if idx.size(1) > 1:
                # 在RNN模式下，我们一次只处理一个token
                logger.warning(f"在RNN模式下收到长度为{idx.size(1)}的序列，只处理第一个token")
                idx = idx[:, 0:1]
        
        # 嵌入输入
        x = self.emb(idx)  # [B, 1, C]
        x = self.optimize_tensor(x)
        
        # 通过各层处理
        for i, block in enumerate(self.blocks):
            x, state = block(x, state=state, mode=RWKVMode.RNN)
        
        # 输出层
        x = self.ln_out(x)
        logits = self.head(x)  # [B, 1, vocab_size]
        
        if return_state:
            return logits, state
        else:
            return logits
    
    def get_device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device
    
    @staticmethod
    def clear_gpu_memory(force=False):
        """清理GPU内存"""
        if not torch.cuda.is_available():
            return
        
        # 获取内存使用情况
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        usage_ratio = allocated / total if total > 0 else 0
        
        # 只有当内存使用率超过阈值或强制清理时才执行
        if force or usage_ratio > 0.8:  # 使用0.8作为阈值
            logger.info(f"清理GPU内存 (使用率: {usage_ratio:.2f})")
            gc.collect()
            torch.cuda.empty_cache()
    
    def optimize_tensor(self, tensor, dtype=None, device=None, inplace=False):
        """优化张量的内存使用"""
        if tensor is None:
            return None
            
        # 如果不指定dtype，使用模型的默认dtype
        if dtype is None:
            dtype = self.dtype
            
        # 如果不指定device，使用模型的device
        if device is None:
            device = self.get_device()
            
        # 如果不是inplace操作，创建新张量
        if not inplace:
            return tensor.to(dtype=dtype, device=device)
            
        # inplace操作
        tensor.data = tensor.data.to(dtype=dtype, device=device)
        return tensor


# 工具函数

def convert_transformer_to_rnn_state(model, last_tokens, state=None):
    """
    将Transformer模式下处理的上下文转换为RNN状态
    用于在训练后切换到推理模式时保持上下文
    
    参数:
        model: RWKV_Dual模型实例
        last_tokens: 最后的token序列，用于初始化状态
        state: 可选的现有状态，如果为None则创建新状态
        
    返回:
        初始化的RNN状态
    """
    # 保存当前模式
    original_mode = model.mode
    
    # 切换到Transformer模式
    model.set_mode(RWKVMode.TRANSFORMER)
    
    # 创建状态（如果未提供）
    if state is None:
        state = model.create_state(batch_size=1)
    
    # 确保last_tokens是张量
    if not isinstance(last_tokens, torch.Tensor):
        last_tokens = torch.tensor(last_tokens, device=model.get_device(), dtype=torch.long)
    
    # 如果是单个token，添加批次维度
    if last_tokens.dim() == 1:
        last_tokens = last_tokens.unsqueeze(0)
    
    # 运行模型获取v_first
    with torch.no_grad():
        _, v_first = model._forward_transformer(last_tokens, return_v_first=True)
    
    # 保存v_first到状态
    state.v_first = v_first[:, -1:, :].detach()
    
    # 切换到RNN模式
    model.set_mode(RWKVMode.RNN)
    
    # 逐个处理最后的tokens来初始化RNN状态
    with torch.no_grad():
        for i in range(last_tokens.size(1)):
            token = last_tokens[:, i:i+1]
            _, state = model._forward_rnn(token, state, return_state=True)
    
    # 恢复原始模式
    model.set_mode(original_mode)
    
    return state


def generate_with_streaming(
    model,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
    repetition_penalty=1.0,
    streaming_callback=None,
    **kwargs
):
    """
    使用RNN模式流式生成文本
    
    参数:
        model: RWKV_Dual模型实例
        prompt: 提示文本或token ID列表
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: 核采样的概率阈值
        top_k: 采样时考虑的最高概率token数
        repetition_penalty: 重复惩罚系数
        streaming_callback: 流式回调函数，接收新生成的token
        **kwargs: 传递给模型
        
    返回:
        generated_text: 生成的文本
    """
    # 保存原始模式
    original_mode = model.get_mode()
    
    # 确保模型处于RNN模式
    model.set_mode(RWKVMode.RNN)
    
    # 处理输入提示
    if isinstance(prompt, str):
        # 如果提示是字符串，使用tokenizer转换为token IDs
        if hasattr(model, 'tokenizer'):
            input_ids = model.tokenizer.encode(prompt)
        else:
            raise ValueError("模型没有tokenizer属性，请提供token ID列表作为prompt")
    else:
        # 否则假设prompt已经是token ID列表
        input_ids = prompt
    
    # 转换为张量
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.get_device())
    
    # 初始化状态
    state = init_state_with_past(model, input_ids)
    
    # 初始化生成结果
    generated_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    
    # 生成新token
    for _ in range(max_new_tokens):
        # 获取最后一个token
        current_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=model.get_device())
        
        # 前向传播
        with torch.no_grad():
            logits, state = model._forward_rnn(current_token, state, return_state=True)
        
        # 应用温度
        if temperature > 0:
            logits = logits / temperature
        
        # 应用重复惩罚
        if repetition_penalty != 1.0:
            for id in set(generated_ids):
                logits[0, 0, id] /= repetition_penalty
        
        # 应用top_k采样
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # 应用top_p采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs[0, 0], 1)[0]
        
        # 添加到生成结果
        generated_ids.append(next_token.item())
        
        # 调用流式回调（如果提供）
        if streaming_callback is not None:
            streaming_callback(next_token.item())
        
        # 检查是否生成了结束标记
        if hasattr(model, 'tokenizer') and next_token.item() == model.tokenizer.eos_token_id:
            break
    
    # 恢复原始模式
    model.set_mode(original_mode)
    
    # 将生成的token转换为文本
    if hasattr(model, 'tokenizer'):
        generated_text = model.tokenizer.decode(generated_ids)
        return generated_text
    else:
        # 如果没有tokenizer，返回token ID列表
        return generated_ids

def init_state_with_past(model, past_tokens, state=None):
    """
    使用过去的token序列初始化RNN状态
    
    参数:
        model: RWKV_Dual模型实例
        past_tokens: 过去的token序列，用于初始化状态
        state: 可选的现有状态，如果为None则创建新状态
        
    返回:
        初始化的RNN状态
    """
    # 保存当前模式
    original_mode = model.mode
    
    # 确保输入是张量
    if not isinstance(past_tokens, torch.Tensor):
        past_tokens = torch.tensor(past_tokens, device=model.get_device(), dtype=torch.long)
    
    # 如果是单个token，添加批次维度
    if past_tokens.dim() == 1:
        past_tokens = past_tokens.unsqueeze(0)
    
    # 创建状态（如果未提供）
    if state is None:
        state = model.create_state(batch_size=past_tokens.size(0))
    
    # 切换到RNN模式
    model.set_mode(RWKVMode.RNN)
    
    # 逐个处理token来初始化状态
    with torch.no_grad():
        for i in range(past_tokens.size(1)):
            token = past_tokens[:, i:i+1]
            _, state = model._forward_rnn(token, state, return_state=True)
    
    # 恢复原始模式
    model.set_mode(original_mode)
    
    return state

def get_logits(model, tokens, state=None):
    """
    获取给定tokens的logits输出
    
    参数:
        model: RWKV_Dual模型实例
        tokens: 输入token序列
        state: RNN模式下的状态（可选）
        
    返回:
        logits: 输出logits
        state: 更新的状态（如果在RNN模式下）
    """
    # 保存当前模式
    original_mode = model.mode
    
    # 根据模式选择不同的处理方式
    if model.mode == RWKVMode.TRANSFORMER:
        logits = model._forward_transformer(tokens, return_v_first=False)
        return logits
    else:
        # 确保有状态
        if state is None:
            state = model.create_state()
        
        # 处理单个token或token序列
        if isinstance(tokens, int):
            tokens = torch.tensor([[tokens]], device=model.get_device(), dtype=torch.long)
        elif isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], int):
            # 处理token ID列表
            tokens = torch.tensor([tokens], device=model.get_device(), dtype=torch.long)
        
        # 如果是张量但维度不正确，调整维度
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
        
        # 在RNN模式下，逐个处理token
        if tokens.size(1) > 1:
            # 初始化状态
            state = init_state_with_past(model, tokens[:, :-1], state)
            # 处理最后一个token
            logits, state = model._forward_rnn(tokens[:, -1:], state, return_state=True)
        else:
            # 只有一个token
            logits, state = model._forward_rnn(tokens, state, return_state=True)
        
        return logits, state

def sample_logits(logits, temperature=1.0, top_p=0.9, top_k=0):
    """
    从logits中采样下一个token
    
    参数:
        logits: 模型输出的logits
        temperature: 采样温度
        top_p: 核采样的概率阈值
        top_k: 采样时考虑的最高概率token数
        
    返回:
        采样的token ID
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # 确保logits是正确的形状
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    
    # 应用温度
    if temperature > 0:
        logits = logits / temperature
    
    # 应用top_k采样
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    
    # 应用top_p采样
    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    # 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 采样
    next_token = torch.multinomial(probs, 1).squeeze(-1)
    
    return next_token

def generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.85,
    top_k=0,
    repetition_penalty=1.0,
    alpha_presence=0.2,
    alpha_frequency=0.2,
    alpha_decay=0.996,
    token_ban=[],
    token_stop=[11],  # 默认使用11作为停止token
    streaming_callback=None,
    tokenizer=None,
    **kwargs
):
    """
    使用模型生成文本
    
    参数:
        model: RWKV_Dual模型实例
        prompt: 提示文本或token ID列表
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: 核采样的概率阈值
        top_k: 采样时考虑的最高概率token数
        repetition_penalty: 重复惩罚系数（已弃用，使用alpha参数代替）
        alpha_presence: 基础重复惩罚系数
        alpha_frequency: 频率重复惩罚系数
        alpha_decay: 重复惩罚衰减系数
        token_ban: 禁止生成的token列表
        token_stop: 停止生成的token列表
        streaming_callback: 流式回调函数，接收新生成的token
        **kwargs: 传递给模型的其他参数
        
    返回:
        generated_tokens: 生成的token ID列表
    """
    # 保存原始模式
    original_mode = model.mode
    
    # 确保模型处于RNN模式
    model.set_mode(RWKVMode.RNN)
    
    # 处理输入提示
    if isinstance(prompt, str):
        # 如果提示是字符串，使用tokenizer转换为token IDs
        if hasattr(model, 'tokenizer'):
            input_ids = model.tokenizer.encode(prompt)
        else:
            raise ValueError("模型没有tokenizer属性，请提供token ID列表作为prompt")
    else:
        # 否则假设prompt已经是token ID列表
        input_ids = prompt
    
    # 转换为张量
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.get_device())
    
    # 初始化状态
    state = init_state_with_past(model, input_ids)
    
    # 初始化生成结果
    generated_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    
    # 初始化token出现次数统计
    occurrence = {}
    
    # 生成新token
    for _ in range(max_new_tokens):
        # 获取最后一个token
        current_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=model.get_device())
        
        # 前向传播
        with torch.no_grad():
            logits, state = model._forward_rnn(current_token, state, return_state=True)
        
        # 获取最后一个位置的logits
        last_logits = logits[0, 0, :]
        
        # 应用重复惩罚
        for token in occurrence:
            occurrence[token] *= alpha_decay
            last_logits[token] -= alpha_presence + occurrence[token] * alpha_frequency
        
        # 应用token黑名单
        for token in token_ban:
            last_logits[token] = -1e38
        
        # 采样下一个token
        if temperature == 0:
            # 贪婪解码
            next_token_id = torch.argmax(last_logits).item()
        else:
            # 使用functions.py中的sample_logits函数
            next_token_id = sample_logits(last_logits, temperature, top_p, top_k)
        
        # 更新token出现统计
        if 49 <= next_token_id <= 58:  # 数字token特殊处理
            pass
        elif next_token_id not in occurrence:
            occurrence[next_token_id] = 1
        else:
            occurrence[next_token_id] += 1
        
        # 添加到生成结果
        next_token_id = next_token_id.item()
        generated_ids.append(next_token_id)
        
        # 调用流式回调（如果提供）
        if streaming_callback is not None:
            streaming_callback(next_token_id)
        else:
            print(tokenizer.decode([next_token_id]), end="")
        
        # 检查是否生成了停止token
        if next_token_id in token_stop:
            break
        
        # 定期清理GPU内存
        if _ % 50 == 0:
            model.clear_gpu_memory(force=False)
    
    # 恢复原始模式
    model.set_mode(original_mode)
    
    return generated_ids

def batch_decode(model, token_ids, skip_special_tokens=True):
    """
    将token ID批量解码为文本
    
    参数:
        model: RWKV_Dual模型实例
        token_ids: token ID列表或批次
        skip_special_tokens: 是否跳过特殊token
        
    返回:
        decoded_texts: 解码后的文本列表
    """
    if not hasattr(model, 'tokenizer'):
        raise ValueError("模型没有tokenizer属性，无法解码token")
    
    # 处理单个ID列表
    if isinstance(token_ids, list) and (not token_ids or isinstance(token_ids[0], int)):
        return model.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    # 处理批次
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    
    # 批量解码
    return [model.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]

def save_pretrained(model, save_directory, save_config=True):
    """
    保存预训练模型
    
    参数:
        model: RWKV_Dual模型实例
        save_directory: 保存目录
        save_config: 是否保存配置
        
    返回:
        None
    """
    import os
    import json
    
    # 创建目录
    os.makedirs(save_directory, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # 保存配置
    if save_config:
        config = {
            "model_type": "RWKV_Dual",
            "n_layer": model.n_layer,
            "n_embd": model.n_embd,
            "vocab_size": model.vocab_size,
            "ctx_len": model.ctx_len,
            "head_size": model.blocks[0].att.head_size if model.blocks else 64,
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return None

def from_pretrained(cls, pretrained_model_path, **kwargs):
    """
    从预训练模型加载
    
    参数:
        cls: RWKV_Dual类
        pretrained_model_path: 预训练模型路径
        **kwargs: 其他参数
        
    返回:
        model: 加载的模型实例
    """
    import os
    import json
    
    # 加载配置
    config_path = os.path.join(pretrained_model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 更新配置
        for key, value in config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    # 设置模型路径
    model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = pretrained_model_path  # 直接使用提供的路径
    
    # 创建模型
    kwargs['load_model'] = model_path
    model = cls(**kwargs)
    
    return model

# 将方法添加到RWKV_Dual类
RWKV_Dual.init_state_with_past = init_state_with_past
RWKV_Dual.get_logits = get_logits
RWKV_Dual.sample_logits = staticmethod(sample_logits)
RWKV_Dual.generate = generate
RWKV_Dual.batch_decode = batch_decode
RWKV_Dual.save_pretrained = save_pretrained
RWKV_Dual.from_pretrained = classmethod(from_pretrained)