import os

os.environ["WORKING_MODE"] = "train_service"

from config import global_config

import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader
from RWKV.functions import train_forward

train_config = global_config.train_service_config
os.environ["RWKV_HEAD_SIZE_A"] = str(
            global_config.pretrain_script_config.model.head_size
        )

from RWKV.v7.model import RWKV

# 修改 SimpleDataset 类
class SimpleDataset(Dataset):
    def __init__(self, size=1000, seq_length=512):
        self.size = size
        # 确保 seq_length 是 CHUNK_LEN (24) 的整数倍
        self.seq_length = (seq_length // 24) * 24   # 向下取整到最近的24的倍数
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机输入和目标
        inputs = torch.randint(0, 50257, (self.seq_length,))
        targets = torch.randint(0, 50257, (self.seq_length,))
        return {'input_ids': inputs, 'labels': targets}

# DeepSpeed配置
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
        "allgather_bucket_size": 2e3,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e3,
        "contiguous_gradients": True
      },
      "gradient_accumulation_steps": 1,
      "gradient_clipping": 1,
      "train_micro_batch_size_per_gpu": 2
    }

# 初始化模型和数据加载器时使用调整后的序列长度
model = RWKV(train_config)

optimizer, lr_scheduler = model.get_optim_groups()

# 使用能被24整除的序列长度
seq_length = (train_config.model.ctx_len // 24) * 24 
dataset = SimpleDataset(seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler
)

# 训练循环
def train():
    model_engine.train()
    for epoch in range(2):  # 训练2个epoch作为示例
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input_ids'].cuda()
            targets = batch['labels'].cuda()
            
            # Create masks (all ones since we're using random data)
            batch_masks = torch.ones_like(inputs, dtype=torch.float32).cuda()
            print("===input shape===", inputs.shape, batch_masks.shape)
            batch_masks = batch_masks
            loss = model_engine(inputs, targets, batch_masks)
            
            model_engine.backward(loss)
            model_engine.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()