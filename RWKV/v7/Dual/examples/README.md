# RWKV v7 双模式设计

RWKV v7 双模态设计是一种创新的架构，允许模型在两种不同的操作模式之间无缝切换：

1. **Transformer模式**：用于并行训练和批处理推理
2. **RNN模式**：用于高效的序列生成和推理

## 特性

- **统一架构**：相同的模型权重可以在两种模式下使用
- **高效训练**：利用Transformer模式进行并行训练
- **快速推理**：使用RNN模式进行高效的序列生成
- **内存优化**：RNN模式显著减少推理时的内存占用
- **状态转换**：支持从Transformer状态到RNN状态的转换，实现混合推理

## 使用方法

### 模型初始化

```python
from RWKV.v7.Dual.model import RWKV_Dual
from RWKV.v7.Dual.mode import RWKVMode

# 创建模型
model = RWKV_Dual(
    n_embd=768,       # 嵌入维度
    n_layer=12,       # 层数
    vocab_size=50277, # 词汇表大小
    ctx_len=1024,     # 上下文长度
    mode=RWKVMode.TRANSFORMER  # 默认模式
)

# 加载预训练权重
model.load_state_dict(torch.load("path/to/weights.pth"))
```

### 切换模式

```python
# 切换到Transformer模式（用于训练或批处理推理）
model.set_mode(RWKVMode.TRANSFORMER)

# 切换到RNN模式（用于序列生成）
model.set_mode(RWKVMode.RNN)
```

### Transformer模式下的前向传播

```python
# 批处理输入
batch_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# Transformer模式下的前向传播
outputs, v_first = model(batch_tokens, return_v_first=True)
```

### RNN模式下的前向传播

```python
# 创建初始状态
state = model.create_state(batch_size=1)

# 逐个处理token
for token in input_sequence:
    token_tensor = torch.tensor([[token]])
    output, state = model(token_tensor, state=state)
```

### 状态转换与混合推理

```python
from RWKV.v7.Dual.model import init_state_with_past

# 使用Transformer模式处理上下文
model.set_mode(RWKVMode.TRANSFORMER)
context_output, _ = model(context_tokens)

# 将上下文转换为RNN状态
state = init_state_with_past(model, context_tokens)

# 切换到RNN模式进行生成
model.set_mode(RWKVMode.RNN)
for i in range(num_tokens_to_generate):
    token_tensor = torch.tensor([[next_token]])
    output, state = model(token_tensor, state=state)
    # 处理输出，选择下一个token
```

## 最佳实践

### 训练

1. **使用Transformer模式**：训练时始终使用Transformer模式以利用并行计算
2. **梯度检查点**：对于大型模型，启用梯度检查点以减少内存使用
3. **混合精度训练**：使用bfloat16或float16进行训练以提高性能

```python
# 启用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

with autocast():
    outputs, _ = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 推理

1. **上下文处理**：对于长上下文，使用Transformer模式批量处理
2. **序列生成**：生成时切换到RNN模式以减少内存使用并提高速度
3. **状态缓存**：缓存和重用RNN状态以避免重复计算

```python
# 高效推理示例
def generate_efficiently(model, prompt, max_length):
    # 处理提示
    model.set_mode(RWKVMode.TRANSFORMER)
    prompt_tokens = tokenize(prompt)
    
    # 初始化RNN状态
    state = init_state_with_past(model, prompt_tokens)
    
    # 切换到RNN模式进行生成
    model.set_mode(RWKVMode.RNN)
    generated = []
    
    for _ in range(max_length):
        if not generated:
            # 使用提示的最后一个token
            token = prompt_tokens[0, -1:].clone()
        else:
            # 使用上一步生成的token
            token = torch.tensor([[generated[-1]]], device=model.get_device())
        
        # 前向传播
        logits, state = model(token, state=state)
        
        # 采样下一个token
        next_token = sample_token(logits)
        generated.append(next_token)
    
    return decode(generated)
```

### 内存优化

1. **状态分离**：使用`state.detach()`分离计算图以避免内存泄漏
2. **清理缓存**：定期使用`torch.cuda.empty_cache()`清理GPU内存
3. **批量大小控制**：在RNN模式下使用较大的批量大小，但避免过大

```python
# 内存优化示例
def optimize_memory():
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 分离状态计算图
    state = state.detach()
    
    # 监控内存使用
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"当前GPU内存使用: {allocated:.2f} MB")
```

## 高级功能

### 模型保存与加载

```python
# 保存模型
model.save_pretrained("path/to/save")

# 加载模型
from RWKV.v7.Dual.model import RWKV_Dual
model = RWKV_Dual.from_pretrained("path/to/save")
```

### 批量解码

```python
# 批量解码token序列
texts = model.batch_decode(token_sequences)
```

### 流式生成

```python
# 流式生成文本
def on_token(token, text):
    print(text, end="", flush=True)

generated_text = model.generate_with_streaming(
    prompt="Once upon a time",
    max_new_tokens=100,
    streaming_callback=on_token
)
```

## 性能比较

| 模式 | 训练速度 | 推理速度 | 内存使用 | 适用场景 |
|------|----------|----------|----------|----------|
| Transformer | 快 | 中等 | 高 | 训练、批处理推理 |
| RNN | 不适用 | 快 | 低 | 序列生成、长文本推理 |

## 故障排除

1. **数据类型不匹配**：确保所有输入张量都使用相同的数据类型（推荐bfloat16）
2. **CUDA错误**：如果遇到CUDA错误，尝试回退到CPU实现或减小批量大小
3. **内存不足**：减小批量大小或上下文长度，或切换到RNN模式
4. **精度问题**：如果RNN和Transformer模式的输出差异过大，检查数值精度和状态转换

## 参考资源

- [RWKV官方仓库](https://github.com/BlinkDL/RWKV-LM)
- [RWKV论文](https://arxiv.org/abs/2305.13048)