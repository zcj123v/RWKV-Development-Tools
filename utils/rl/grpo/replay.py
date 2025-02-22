import torch
import torch.nn.functional as F
from dataclasses import dataclass, fields
from typing import Optional, Self, List

from utils.rl.grpo.functions import zero_pad_sequences


@dataclass
class ExperienceHist:
    history_tokens: torch.Tensor  # 完整输出的tokens
    action_log_probs: torch.Tensor  # 完整输出的logp
    log_probs_ref: torch.Tensor  # 参考模型的完整输出logp
    rewards: Optional[torch.Tensor]  # 一个batch的rewards
    advantages: Optional[torch.Tensor]  # 归一化的rewards
    action_mask: torch.Tensor  # 为0则不计算loss
    kl: Optional[torch.Tensor] = None  # 计算的kl

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return ExperienceHist(**members)

    def unbind(self) -> Self:
        batch_size = self.sequences.size(0)
        batch_data = [{} for _ in range(batch_size)]
        keys = (
            "history_tokens",
            "action_log_probs",
            "log_probs_ref",
            "rewards",
            "advantages",
            "action_mask",
        )
        for key in keys:
            value = getattr(self, key)
            if value is None:
                vals = [None] * batch_size
            else:
                vals = torch.unbind(value)
            assert batch_size == len(vals)
            for i, v in enumerate(vals):
                batch_data[i][key] = v

        return [ExperienceHist(**data) for data in batch_data]

    def __add__(self, other) -> List[Self]:
        batch_data = {}
        keys = (
            "history_tokens",
            "action_log_probs",
            "log_probs_ref",
            "rewards",
            "advantages",
            "action_mask",
        )
        for key in keys:
            sv = getattr(self, key)
            v = getattr(other, key)
            if v is not None:
                data = zero_pad_sequences([sv, v], "left")
            else:
                data = None
            batch_data[key] = data
        return ExperienceHist(**batch_data)

    @staticmethod
    def gather(items: list[Self]) -> Self:
        batch_data = {}
        keys = (
            "history_tokens",
            "action_log_probs",
            "log_probs_ref",
            "rewards",
            "advantages",
            "action_mask",
        )
        for key in keys:
            vals = [getattr(item, key) for item in items]
            if all(v is not None for v in vals):
                vals: List[torch.Tensor]
                vv = []
                for v in vals:
                    vv += torch.unbind(v)
                data = zero_pad_sequences(vv, "left")
            else:
                data = None
            batch_data[key] = data
        return ExperienceHist(**batch_data)


class ReplaySlidingWindow:
    def __init__(self, window_size: int = 0):
        self.window_size = window_size
        self.buffer = []

    def add(self, item: ExperienceHist):
        self.buffer.append(item)
        if len(self.buffer) > self.window_size and self.window_size > 0:
            self.buffer = self.buffer[-self.window_size :]

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]
