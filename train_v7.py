import os

os.environ["WORKING_MODE"] = "train_service"

from gevent import monkey

monkey.patch_all()
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

# 创建一个简单的测试数据集
class SimpleDataset(Dataset):
    def __init__(self, size=1000, seq_length=512):
        self.size = size
        self.seq_length = seq_length
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机输入和目标
        inputs = torch.randint(0, 50257, (self.seq_length,))
        targets = torch.randint(0, 50257, (self.seq_length,))
        return {'input_ids': inputs, 'labels': targets}

# DeepSpeed配置
ds_config = {
    "train_batch_size": 2,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}

# 初始化模型
model = RWKV(train_config)
optimizer, lr_scheduler = model.get_optim_groups()
# 准备数据
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

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
            batch_masks = torch.ones_like(inputs, dtype=torch.float32)
            
            # Use train_forward instead of manual loss calculation
            loss, _ = train_forward(model_engine, inputs, batch_masks)
            
            model_engine.backward(loss)
            model_engine.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()