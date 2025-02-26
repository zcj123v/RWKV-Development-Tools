import os
os.environ["WORKING_MODE"] = "train_service"

import torch
import deepspeed
import gc
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from config import global_config

# Set head size in environment
train_config = global_config.train_service_config
os.environ["RWKV_HEAD_SIZE_A"] = str(global_config.pretrain_script_config.model.head_size)

# Import model after setting environment variables
from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.model import RWKV_Dual
from RWKV.v7.Dual.optimizer import RWKVOptimizer

# Initialize tokenizer
tokenizer = global_config.tokenizer_eval

# Load eval text for training examples
with open("/home/neromous/MachineLr/RWKV-Development-Tools/resources/docs/RWKV7结构解析.md", "r") as f:
    eval_text = f.read()

# Tokenize and ensure length is divisible by CHUNK_LEN (24)
eval_tokens = tokenizer.encode(eval_text)
eval_tokens_len = (len(eval_tokens) // 24) * 24
eval_tokens = eval_tokens[:eval_tokens_len]

class RWKVDataset(Dataset):
    def __init__(self, size=5, seq_length=512):
        self.size = size
        # Important: ensure divisibility by CHUNK_LEN (24)
        self.seq_length = (seq_length // 24) * 24
        self.eval_tokens = eval_tokens
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Randomly select a starting position from eval_tokens
        if len(self.eval_tokens) >= self.seq_length + 1:
            start_idx = torch.randint(0, len(self.eval_tokens) - self.seq_length - 1, (1,)).item()
            inputs = torch.tensor(self.eval_tokens[start_idx:start_idx + self.seq_length])
            targets = torch.tensor(self.eval_tokens[start_idx + 1:start_idx + self.seq_length + 1])
        else:
            # If eval_tokens isn't long enough, use random data
            # Make sure both have lengths divisible by CHUNK_LEN (24)
            inputs = torch.randint(0, 50257, (self.seq_length,))
            # Since we're shifting by 1, pad an extra token at the end
            shifted = torch.roll(inputs, shifts=-1)
            targets = shifted.clone()
            targets[-1] = torch.randint(0, 50257, (1,))
        return {'input_ids': inputs, 'labels': targets}

# DeepSpeed configuration
ds_config = {
    "bfloat16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2
}

def test_inference(model, tokenizer):
    """
    使用模型的generate方法测试推理能力
    """
    print("\n===== 开始推理测试 =====")
    
    # 将模型切换到RNN模式
    model.set_mode(RWKVMode.RNN)
    
    # 准备提示文本
    prompt = "RWKV是一种创新的语言模型架构，它结合了"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"提示: '{prompt}'")
    print("生成中...")
    
    # 使用模型的generate方法进行推理
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            stop_sequences=[[0]]  # 以0作为停止token
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_tokens[len(prompt_tokens):])
    print("\n===== 生成结果 =====")
    print(prompt + generated_text)
    print("=====================")
    
    # 切换回Transformer模式用于训练
    model.set_mode(RWKVMode.TRANSFORMER)
    
    return generated_text

def create_optimizer(model, config):
    """
    创建优化器和学习率调度器，使用 RWKVOptimizer 工具类
    """
    # 使用 RWKVOptimizer 创建优化器和学习率调度器
    optimizer, lr_scheduler = RWKVOptimizer.create_optimizer(
        model=model,
        lr_init=config.train.lr_init if hasattr(config.train, 'lr_init') else 5e-4,
        lr_final=config.train.lr_final if hasattr(config.train, 'lr_final') else 1e-5,
        beta1=config.train.beta1 if hasattr(config.train, 'beta1') else 0.9,
        beta2=config.train.beta2 if hasattr(config.train, 'beta2') else 0.99,
        adam_eps=config.train.adam_eps if hasattr(config.train, 'adam_eps') else 1e-8,
        weight_decay=config.train.weight_decay if hasattr(config.train, 'weight_decay') else 0.01,
        warmup_steps=config.train.warmup_steps if hasattr(config.train, 'warmup_steps') else 1000,
        # 可选参数
        layerwise_lr=config.train.layerwise_lr if hasattr(config.train, 'layerwise_lr') else 0.0,
        my_pile_stage=0,  # 默认为0
        adamw_mode=True   # 使用AdamW模式
    )
    
    return optimizer, lr_scheduler

def train():
    print("Initializing RWKV Dual model...")
    
    # Make sure ctx_len is divisible by CHUNK_LEN (24)
    ctx_len = train_config.model.ctx_len if hasattr(train_config.model, 'ctx_len') else 1024
    ctx_len = (ctx_len // 24) * 24
    
    # Create model instance
    model = RWKV_Dual(
        # Model architecture parameters
        n_embd=train_config.model.n_embd if hasattr(train_config.model, 'n_embd') else -1,
        n_layer=train_config.model.n_layer if hasattr(train_config.model, 'n_layer') else -1,
        vocab_size=train_config.model.vocab_size if hasattr(train_config.model, 'vocab_size') else -1,
        head_size=train_config.model.head_size if hasattr(train_config.model, 'head_size') else 64,
        head_size_divisor=train_config.model.head_size_divisor if hasattr(train_config.model, 'head_size_divisor') else 8,
        ctx_len=ctx_len,
        
        # Model loading parameters  
        load_model=train_config.model.load_model if hasattr(train_config.model, 'load_model') else None,
        dtype=train_config.model.dtype if hasattr(train_config.model, 'dtype') else "bf16",
        
        # Training parameters
        dropout=train_config.train.dropout if hasattr(train_config.train, 'dropout') else 0.0,
        grad_cp=1,
        
        # Set to Transformer mode for training
        mode=RWKVMode.TRANSFORMER
    )
    
    # Add tokenizer attribute to model for convenience
    model.tokenizer = tokenizer
    
    print(f"Model initialized with {model.n_layer} layers, {model.n_embd} embedding size")
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer(model, train_config)
    
    # Initialize dataset and dataloader
    # Make sure seq_length is divisible by CHUNK_LEN (24)
    seq_length = (ctx_len // 24) * 24
    batch_size = ds_config["train_micro_batch_size_per_gpu"]
    dataset = RWKVDataset(size=50, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset prepared with sequence length {seq_length}, batch size {batch_size}")
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    test_inference(model_engine.module, tokenizer)

    print("Starting training...")
    # Training loop
    model_engine.train()
    for epoch in range(2):  # 2 epochs as an example
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(model_engine.device)
            targets = batch['labels'].to(model_engine.device)
            
            # Forward pass using the model's forward method directly
            # Add padding to inputs if needed to make length divisible by CHUNK_LEN
            pad_len = (24 - (inputs.size(1) % 24)) % 24
            if pad_len > 0:
                inputs = F.pad(inputs, (0, pad_len), "constant", 0)
                
            # In Transformer mode, v_first is None initially
            logits = model_engine(inputs, v_first=None, return_state=False, return_v_first=False)
            
            # If we padded, remove padding from logits before computing loss
            if pad_len > 0:
                logits = logits[:, :-pad_len, :]
                
            # Calculate cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                reduction='mean'
            )
            
            # Backward pass and optimization
            model_engine.backward(loss)
            model_engine.step()
            
            # Cleanup
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
                # 添加内存监控
                mem_info = RWKVOptimizer.monitor_memory_usage()
                if 'error' not in mem_info:
                    print(f"GPU内存: 已分配={mem_info['allocated_gb']:.2f}GB, "
                          f"总计={mem_info['total_gb']:.2f}GB, "
                          f"使用率={mem_info['utilization']:.2%}")
                    if mem_info['warning']:
                        print("警告: GPU内存使用率过高!")
            
            # Manual memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        
        # Epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # 每个epoch结束后进行推理测试
        print(f"\n执行Epoch {epoch}后的推理测试")
        # 确保模型在推理前处于正确的模式
        model_engine.module.set_mode(RWKVMode.RNN)
        test_inference(model_engine.module, tokenizer)
        # 推理后切换回训练模式
        model_engine.module.set_mode(RWKVMode.TRANSFORMER)
        
        # Save checkpoint after each epoch
        if hasattr(train_config, 'output_dir'):
            save_path = os.path.join(train_config.output_dir, f"rwkv_dual_epoch_{epoch}")
            model_engine.save_checkpoint(save_path)
            print(f"Model checkpoint saved to {save_path}")

if __name__ == "__main__":
    train() 