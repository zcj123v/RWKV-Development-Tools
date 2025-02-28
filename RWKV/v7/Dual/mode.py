"""
RWKV Mode Definitions
--------------------
This module defines the operation modes for RWKV dual architecture.
"""

from enum import Enum

class RWKVMode(Enum):
    """RWKV模型的运行模式枚举"""
    TRANSFORMER = "transformer"  # 并行训练模式
    RNN = "rnn"                  # 序列推理模式 