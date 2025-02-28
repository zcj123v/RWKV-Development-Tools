import torch
import unittest
import logging
import numpy as np
from RWKV.v7.Dual.model import RWKV_Dual
from RWKV.v7.Dual.mode import RWKVMode
from RWKV.v7.Dual.state import RWKVState

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDualMode(unittest.TestCase):
    """测试RWKV双模态实现的一致性"""
    
    @classmethod
    def setUpClass(cls):
        """创建一个小型测试模型"""
        # 使用小型配置以加快测试速度
        cls.n_embd = 128
        cls.n_layer = 2
        cls.vocab_size = 100
        cls.ctx_len = 64
        
        # 创建模型
        cls.model = RWKV_Dual(
            n_embd=cls.n_embd,
            n_layer=cls.n_layer,
            vocab_size=cls.vocab_size,
            ctx_len=cls.ctx_len,
            dtype="bf16"  # 使用bfloat16以匹配CUDA实现
        )
        
        # 如果CUDA可用，将模型移至GPU
        if torch.cuda.is_available():
            cls.model = cls.model.cuda()
            cls.device = "cuda"
        else:
            cls.device = "cpu"
        
        # 设置为评估模式
        cls.model.eval()
        
        logger.info(f"测试模型创建完成: {cls.n_layer}层, {cls.n_embd}维度, 设备: {cls.device}")
    
    def test_single_token_consistency(self):
        """测试单个token在两种模式下的输出一致性"""
        # 创建随机token
        token_id = torch.randint(0, self.vocab_size, (1, 1), device=self.device)
        
        # Transformer模式下的前向传播
        self.model.set_mode(RWKVMode.TRANSFORMER)
        with torch.no_grad():
            transformer_output, _ = self.model(token_id)
        
        # RNN模式下的前向传播
        self.model.set_mode(RWKVMode.RNN)
        state = self.model.create_state()
        with torch.no_grad():
            rnn_output, _ = self.model(token_id, state=state)
        
        # 检查输出形状
        self.assertEqual(transformer_output.shape, rnn_output.shape, 
                         "Transformer和RNN模式的输出形状不一致")
        
        # 检查输出值的一致性（允许一定的数值误差）
        max_diff = torch.max(torch.abs(transformer_output - rnn_output)).item()
        logger.info(f"单个token测试 - 最大差异: {max_diff:.6f}")
        self.assertLess(max_diff, 1e-3, "Transformer和RNN模式的输出值差异过大")
    
    def test_sequence_consistency(self):
        """测试序列在两种模式下的输出一致性"""
        # 创建随机token序列
        seq_len = 16
        token_ids = torch.randint(0, self.vocab_size, (1, seq_len), device=self.device)
        
        # Transformer模式下的前向传播
        self.model.set_mode(RWKVMode.TRANSFORMER)
        with torch.no_grad():
            transformer_output, _ = self.model(token_ids)
        
        # RNN模式下的前向传播（逐个token处理）
        self.model.set_mode(RWKVMode.RNN)
        state = self.model.create_state()
        rnn_outputs = []
        
        with torch.no_grad():
            for i in range(seq_len):
                token = token_ids[:, i:i+1]
                output, state = self.model(token, state=state)
                rnn_outputs.append(output)
        
        # 将RNN输出连接成序列
        rnn_output = torch.cat(rnn_outputs, dim=1)
        
        # 检查输出形状
        self.assertEqual(transformer_output.shape, rnn_output.shape, 
                         "Transformer和RNN模式的序列输出形状不一致")
        
        # 检查输出值的一致性（允许一定的数值误差）
        max_diff = torch.max(torch.abs(transformer_output - rnn_output)).item()
        logger.info(f"序列测试 - 最大差异: {max_diff:.6f}")
        self.assertLess(max_diff, 1e-2, "Transformer和RNN模式的序列输出值差异过大")
    
    def test_state_conversion(self):
        """测试从Transformer状态转换到RNN状态的一致性"""
        # 创建随机token序列
        seq_len = 8
        token_ids = torch.randint(0, self.vocab_size, (1, seq_len), device=self.device)
        
        # 使用init_state_with_past初始化状态
        from RWKV.v7.Dual.model import init_state_with_past
        
        # 先用Transformer模式处理前半部分序列
        self.model.set_mode(RWKVMode.TRANSFORMER)
        with torch.no_grad():
            first_half = token_ids[:, :seq_len//2]
            transformer_output, v_first = self.model(first_half, return_v_first=True)
        
        # 将状态转换为RNN状态
        state = init_state_with_past(self.model, first_half)
        
        # 用RNN模式处理后半部分序列
        self.model.set_mode(RWKVMode.RNN)
        rnn_outputs = []
        
        with torch.no_grad():
            for i in range(seq_len//2, seq_len):
                token = token_ids[:, i:i+1]
                output, state = self.model(token, state=state)
                rnn_outputs.append(output)
        
        # 将RNN输出连接成序列
        rnn_output = torch.cat(rnn_outputs, dim=1)
        
        # 用Transformer模式处理整个序列作为参考
        self.model.set_mode(RWKVMode.TRANSFORMER)
        with torch.no_grad():
            full_transformer_output, _ = self.model(token_ids)
        
        # 提取后半部分的Transformer输出
        transformer_second_half = full_transformer_output[:, seq_len//2:, :]
        
        # 检查输出形状
        self.assertEqual(transformer_second_half.shape, rnn_output.shape, 
                         "状态转换后的输出形状不一致")
        
        # 检查输出值的一致性（允许一定的数值误差）
        max_diff = torch.max(torch.abs(transformer_second_half - rnn_output)).item()
        logger.info(f"状态转换测试 - 最大差异: {max_diff:.6f}")
        self.assertLess(max_diff, 1e-2, "状态转换后的输出值差异过大")
    
    def test_memory_efficiency(self):
        """测试RNN模式的内存效率"""
        if not torch.cuda.is_available():
            self.skipTest("此测试需要CUDA支持")
        
        # 创建长序列
        seq_len = 128
        token_ids = torch.randint(0, self.vocab_size, (1, seq_len), device=self.device)
        
        # 记录Transformer模式下的内存使用
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.model.set_mode(RWKVMode.TRANSFORMER)
        with torch.no_grad():
            _ = self.model(token_ids)
        
        transformer_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # 记录RNN模式下的内存使用
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.model.set_mode(RWKVMode.RNN)
        state = self.model.create_state()
        
        with torch.no_grad():
            for i in range(seq_len):
                token = token_ids[:, i:i+1]
                _, state = self.model(token, state=state)
        
        rnn_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # 检查RNN模式是否更节省内存
        logger.info(f"内存使用 - Transformer: {transformer_memory:.2f}MB, RNN: {rnn_memory:.2f}MB")
        self.assertLess(rnn_memory, transformer_memory, "RNN模式应该比Transformer模式更节省内存")

if __name__ == "__main__":
    unittest.main() 