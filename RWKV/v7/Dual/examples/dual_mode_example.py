#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RWKV v7 双模态示例脚本
展示如何在训练和推理中切换Transformer和RNN模式
"""

import os
import torch
import argparse
import logging
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from RWKV.v7.Dual.model import RWKV_Dual, init_state_with_past
from RWKV.v7.Dual.mode import RWKVMode

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            # 将长文本分割成多个训练样本
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def train_transformer_mode(model, dataloader, optimizer, device, epochs=1, fp16=True):
    """使用Transformer模式训练模型"""
    logger.info("开始使用Transformer模式训练...")
    
    # 设置为Transformer模式
    model.set_mode(RWKVMode.TRANSFORMER)
    model.train()
    
    # 使用混合精度训练
    scaler = GradScaler() if fp16 else None
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if fp16:
                with autocast():
                    logits, _ = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                
                # 使用混合精度训练
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
    
    logger.info("训练完成")
    return model

def generate_text_hybrid_mode(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """使用混合模式生成文本（Transformer处理上下文，RNN生成）"""
    logger.info(f"使用混合模式生成文本，提示: '{prompt}'")
    
    # 编码提示
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=model.get_device())
    
    # 使用Transformer模式处理上下文
    model.set_mode(RWKVMode.TRANSFORMER)
    model.eval()
    
    with torch.no_grad():
        # 处理上下文
        _, v_first = model(prompt_tensor, return_v_first=True)
        
        # 初始化RNN状态
        state = init_state_with_past(model, prompt_tensor)
        
        # 切换到RNN模式
        model.set_mode(RWKVMode.RNN)
        
        # 生成新token
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            # 获取最后一个token
            if not generated_tokens:
                token = prompt_tensor[:, -1:]
            else:
                token = torch.tensor([[generated_tokens[-1]]], dtype=torch.long, device=model.get_device())
            
            # 前向传播
            logits, state = model(token, state=state)
            
            # 采样下一个token
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Top-p采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)
            
            # 检查是否生成了结束标记
            if next_token == tokenizer.eos_token_id:
                break
    
    # 解码生成的token
    generated_text = tokenizer.decode(generated_tokens)
    logger.info(f"生成的文本: '{generated_text}'")
    
    return generated_text

def benchmark_modes(model, input_length=512, device="cuda"):
    """比较Transformer和RNN模式的性能"""
    logger.info("开始性能基准测试...")
    
    # 创建随机输入
    random_input = torch.randint(0, 1000, (1, input_length), device=device)
    
    # 测量Transformer模式性能
    model.set_mode(RWKVMode.TRANSFORMER)
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(random_input[:, :10])
    
    # 计时
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        _ = model(random_input)
    end_time.record()
    torch.cuda.synchronize()
    
    transformer_time = start_time.elapsed_time(end_time)
    logger.info(f"Transformer模式处理{input_length}个token耗时: {transformer_time:.2f}ms")
    
    # 测量RNN模式性能
    model.set_mode(RWKVMode.RNN)
    state = model.create_state()
    
    # 预热
    with torch.no_grad():
        for i in range(3):
            _, state = model(random_input[:, i:i+1], state=state)
    
    # 重置状态
    state = model.create_state()
    
    # 计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        for i in range(input_length):
            _, state = model(random_input[:, i:i+1], state=state)
    end_time.record()
    torch.cuda.synchronize()
    
    rnn_time = start_time.elapsed_time(end_time)
    logger.info(f"RNN模式处理{input_length}个token耗时: {rnn_time:.2f}ms")
    
    # 测量内存使用
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.set_mode(RWKVMode.TRANSFORMER)
    with torch.no_grad():
        _ = model(random_input)
    
    transformer_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.set_mode(RWKVMode.RNN)
    state = model.create_state()
    with torch.no_grad():
        for i in range(input_length):
            _, state = model(random_input[:, i:i+1], state=state)
    
    rnn_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    logger.info(f"Transformer模式内存使用: {transformer_memory:.2f}MB")
    logger.info(f"RNN模式内存使用: {rnn_memory:.2f}MB")
    logger.info(f"内存节省: {(transformer_memory - rnn_memory) / transformer_memory * 100:.2f}%")
    
    return {
        "transformer_time": transformer_time,
        "rnn_time": rnn_time,
        "transformer_memory": transformer_memory,
        "rnn_memory": rnn_memory
    }

def main():
    parser = argparse.ArgumentParser(description="RWKV v7 双模态示例")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--mode", type=str, choices=["train", "generate", "benchmark"], default="benchmark", help="运行模式")
    parser.add_argument("--prompt", type=str, default="今天天气真不错，我决定", help="生成模式的提示文本")
    parser.add_argument("--max_tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    args = parser.parse_args()
    
    # 创建或加载模型
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"从 {args.model_path} 加载模型")
        model = RWKV_Dual.from_pretrained(args.model_path)
    else:
        logger.info("创建新模型")
        model = RWKV_Dual(
            n_embd=768,
            n_layer=12,
            vocab_size=50277,
            ctx_len=1024
        )
    
    model = model.to(args.device)
    
    # 根据模式执行不同操作
    if args.mode == "train":
        # 这里应该使用实际的tokenizer和数据集
        # 为了示例，我们使用一个简单的模拟
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except ImportError:
            logger.error("请安装transformers库: pip install transformers")
            return
        
        # 模拟数据
        texts = [
            "这是一个示例文本，用于展示RWKV v7的双模态训练。",
            "RWKV模型可以在Transformer模式和RNN模式之间切换，非常灵活。",
            "Transformer模式适合并行训练，而RNN模式适合高效推理。"
        ]
        
        # 创建数据集和数据加载器
        dataset = SimpleDataset(texts, tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 创建优化器
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # 训练模型
        train_transformer_mode(model, dataloader, optimizer, args.device)
        
        # 保存模型
        if args.model_path:
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            model.save_pretrained(args.model_path)
            logger.info(f"模型已保存到 {args.model_path}")
    
    elif args.mode == "generate":
        # 加载tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except ImportError:
            logger.error("请安装transformers库: pip install transformers")
            return
        
        # 生成文本
        generated_text = generate_text_hybrid_mode(
            model, tokenizer, args.prompt, args.max_tokens
        )
        print(f"\n生成的文本:\n{args.prompt}{generated_text}")
    
    elif args.mode == "benchmark":
        # 运行基准测试
        results = benchmark_modes(model, input_length=512, device=args.device)
        
        # 打印结果摘要
        print("\n性能比较摘要:")
        print(f"Transformer模式处理512个token耗时: {results['transformer_time']:.2f}ms")
        print(f"RNN模式处理512个token耗时: {results['rnn_time']:.2f}ms")
        print(f"Transformer模式内存使用: {results['transformer_memory']:.2f}MB")
        print(f"RNN模式内存使用: {results['rnn_memory']:.2f}MB")
        print(f"内存节省: {(results['transformer_memory'] - results['rnn_memory']) / results['transformer_memory'] * 100:.2f}%")

if __name__ == "__main__":
    main()
