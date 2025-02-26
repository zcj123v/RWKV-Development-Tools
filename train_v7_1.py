import os
os.environ["WORKING_MODE"] = "train_service"

import torch
import deepspeed
import gc
from torch.utils.data import Dataset, DataLoader
from config import global_config

# Set head size in environment
train_config = global_config.train_service_config
os.environ["RWKV_HEAD_SIZE_A"] = str(global_config.pretrain_script_config.model.head_size)

# Import model after setting environment variables
from RWKV.v7.model import RWKV, create_rwkv_model, RWKVOptimizer

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
        # Important: removed +1 to ensure divisibility by CHUNK_LEN (24)
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


def train():
    print("Initializing RWKV model...")
    
    # Make sure ctx_len is divisible by CHUNK_LEN (24)
    ctx_len = train_config.model.ctx_len if hasattr(train_config.model, 'ctx_len') else 1024
    ctx_len = (ctx_len // 24) * 24
    
    # Create model instance using the create_rwkv_model function
    model = create_rwkv_model(
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
    )
    
    print(f"Model initialized with {model.n_layer} layers, {model.n_embd} embedding size")
    
    # Get optimizer and scheduler components directly from the model
    optimizer, lr_scheduler = RWKVOptimizer.create_optimizer(
        model,
        lr_init=train_config.train.lr_init if hasattr(train_config.train, 'lr_init') else 5e-4,
        lr_final=train_config.train.lr_final if hasattr(train_config.train, 'lr_final') else 1e-5,
        beta1=train_config.train.beta1 if hasattr(train_config.train, 'beta1') else 0.9,
        beta2=train_config.train.beta2 if hasattr(train_config.train, 'beta2') else 0.99,
        adam_eps=train_config.train.adam_eps if hasattr(train_config.train, 'adam_eps') else 1e-8,
        weight_decay=train_config.train.weight_decay if hasattr(train_config.train, 'weight_decay') else 0.01,
        layerwise_lr=train_config.train.layerwise_lr if hasattr(train_config.train, 'layerwise_lr') else 0.0,
        my_pile_stage=train_config.train.my_pile_stage if hasattr(train_config.train, 'my_pile_stage') else 0,
        adamw_mode=train_config.train.adamw_mode if hasattr(train_config.train, 'adamw_mode') else True,
        warmup_steps=train_config.train.warmup_steps if hasattr(train_config.train, 'warmup_steps') else 1000,
    )
    
    # Initialize dataset and dataloader
    # Make sure seq_length is divisible by CHUNK_LEN (24)
    seq_length = (ctx_len // 24) * 24
    batch_size = ds_config["train_micro_batch_size_per_gpu"]
    dataset = RWKVDataset(size=50, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset prepared with sequence length {seq_length}, batch size {batch_size}")
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
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
                
            logits, _ = model_engine(inputs)
            
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
                
            # Manual memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        
        # Epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        if hasattr(train_config, 'output_dir'):
            save_path = os.path.join(train_config.output_dir, f"rwkv_epoch_{epoch}.pt")
            model_engine.save_checkpoint(save_path)
            print(f"Model checkpoint saved to {save_path}")
        

        # Test generation after each epoch
        print("\nRunning text generation test...")
        # We need to use the unwrapped model since DeepSpeed wraps it
        model_to_generate = model_engine.module  # 获取原始模型实例
        generated_tokens = model_to_generate.generate_chinese(
            torch.tensor([tokenizer.encode(eval_text[:100])]), 
            max_length=50,
            tokenizer=tokenizer
        )
        generated_text = tokenizer.decode(generated_tokens)
        print(f"Generated text:\n{generated_text}\n")
        # Continue training
        model_engine.train()
if __name__ == "__main__":
    from torch.nn import functional as F
    train()