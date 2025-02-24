import os

os.environ["WORKING_MODE"] = "train_service"

from config import global_config
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader
from RWKV.functions import train_forward
train_config = global_config.train_service_config
os.environ["RWKV_HEAD_SIZE_A"] = str(global_config.pretrain_script_config.model.head_size)
from RWKV.v7.model_raw import RWKV
import gc
 
tokenizer = global_config.tokenizer_eval

with open("/home/neromous/MachineLr/RWKV-Development-Tools/resources/docs/RWKV7结构解析.md", "r") as f:
    eval_text = f.read()

eval_tokens = tokenizer.encode(eval_text)
# 确保eval_tokens长度是24的倍数
eval_tokens_len = (len(eval_tokens) // 24) * 24 +1
eval_tokens = eval_tokens[:eval_tokens_len]

# 修改 SimpleDataset 类
class SimpleDataset(Dataset):
    def __init__(self, size=1000, seq_length=512):
        self.size = size
        # 确保 seq_length 是 CHUNK_LEN (24) 的整数倍
        self.seq_length = (seq_length // 24) * 24+1   # 向下取整到最近的24的倍数
        self.eval_tokens = eval_tokens
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 从eval_tokens中随机选择一个起始位置
        if len(self.eval_tokens) >= self.seq_length + 1:  # +1 因为我们需要额外一个token作为最后的目标
            start_idx = torch.randint(0, len(self.eval_tokens) - self.seq_length - 1, (1,)).item()
            inputs = torch.tensor(self.eval_tokens[start_idx:start_idx + self.seq_length])
            targets = torch.tensor(self.eval_tokens[start_idx + 1:start_idx + self.seq_length + 1])
        else:
            # 如果eval_tokens长度不够,用随机数据填充
            inputs = torch.randint(0, 50257, (self.seq_length,))
            targets = torch.roll(inputs, shifts=-1)  # 向左移动一位
            targets[-1] = torch.randint(0, 50257, (1,))  # 最后一个位置随机填充
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
            batch_masks = batch_masks
            loss, _ = train_forward(model_engine, inputs, None)
            print("===input shape===",loss)

            model_engine.backward(loss)
            model_engine.step()
            gc.collect()
            torch.cuda.empty_cache()
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()