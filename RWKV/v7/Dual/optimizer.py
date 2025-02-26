import gc
import torch
import logging
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from RWKV.v7.Dual.model import RWKV_Dual

# 设置日志记录器
logger = logging.getLogger(__name__)

# 内存监控阈值
MEM_MONITOR_THRESHOLD = 0.9

# Simplify RWKVOptimizer to be a utility class with better integration
class RWKVOptimizer:
    """RWKV模型优化器工具类"""
    
    @staticmethod
    def get_optim_groups(model, layerwise_lr=0.0, weight_decay=0.01, my_pile_stage=0):
        """
        将模型参数分组以应用不同的学习率和权重衰减
        
        参数:
            model: RWKV模型实例
            layerwise_lr: 是否使用分层学习率（0表示禁用）
            weight_decay: 权重衰减系数
            my_pile_stage: Pile数据集的训练阶段
            
        返回:
            参数分组列表，用于优化器初始化
        """
        # 参数分组
        lr_decay = set()  # 需要权重衰减的参数
        lr_1x = set()     # 基础学习率参数
        lr_2x = set()     # 2倍学习率参数
        lr_3x = set()     # 3倍学习率参数
        
        # 根据参数名称进行分类
        for n, p in model.named_parameters():
            # 权重矩阵参数
            if (("_w1" in n) or ("_w2" in n)) and (layerwise_lr > 0):
                lr_1x.add(n)
            # 时间状态参数（需要权重衰减）
            elif (("time_sta" in n) and (weight_decay > 0)):
                lr_decay.add(n)
            # 时间混合参数
            elif (("time_mix" in n) or ("time_maa" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            # 时间衰减参数
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            # 其他时间相关参数
            elif ("time_faaaa" in n) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (layerwise_lr > 0):
                lr_3x.add(n)
            # 二维及以上权重矩阵
            elif (len(p.squeeze().shape) >= 2) and (weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            # 其他所有参数
            else:
                lr_1x.add(n)

        # 创建参数字典
        param_dict = {n: p for n, p in model.named_parameters()}
        
        # 构建优化器分组
        optim_groups = []
        if layerwise_lr > 0:
            if my_pile_stage == 2:
                # Pile阶段2使用特殊的学习率缩放
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                # 标准分层学习率
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            # 不使用分层学习率
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        # 添加需要权重衰减的参数组
        if weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": weight_decay, "my_lr_scale": 1.0}]
            
        return optim_groups
    
    @classmethod
    def create_optimizer(cls, model, lr_init=5e-4, lr_final=1e-5, beta1=0.9, beta2=0.99, 
                           adam_eps=1e-8, weight_decay=0.01, layerwise_lr=0.0, 
                           my_pile_stage=0, adamw_mode=True, warmup_steps=1000):
        """
        创建RWKV模型的优化器和学习率调度器
        
        参数:
            model: RWKV模型实例
            lr_init: 初始学习率
            lr_final: 最终学习率
            beta1, beta2: Adam优化器的beta参数
            adam_eps: Adam优化器的epsilon值
            weight_decay: 权重衰减系数
            layerwise_lr: 是否使用分层学习率
            my_pile_stage: Pile数据集的训练阶段
            adamw_mode: 是否使用AdamW模式（否则使用Adam）
            warmup_steps: 预热步数
            
        返回:
            optimizer: 配置好的优化器
            lr_scheduler: 配置好的学习率调度器
        """
        # 在创建优化器前清理内存
        if hasattr(model, 'clear_gpu_memory'):
            model.clear_gpu_memory(force=True)
        else:
            gc.collect()
            torch.cuda.empty_cache()
            
        # 获取参数分组
        optim_groups = cls.get_optim_groups(model, layerwise_lr, weight_decay, my_pile_stage)
        
        # 创建DeepSpeed CPU Adam优化器
        optimizer = DeepSpeedCPUAdam(
            optim_groups,
            lr=lr_init,
            betas=(beta1, beta2),
            eps=adam_eps,
            adamw_mode=adamw_mode,
            weight_decay=weight_decay,
            amsgrad=False,
            bias_correction=True,
        )
        
        # 创建预热学习率调度器
        lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
            optimizer,
            warmup_min_lr=lr_final,
            warmup_max_lr=lr_init,
            warmup_num_steps=warmup_steps,
            warmup_type="linear",
        )
        
        # 创建优化器后再次清理内存
        if hasattr(model, 'clear_gpu_memory'):
            model.clear_gpu_memory()
        else:
            gc.collect()
            torch.cuda.empty_cache()
            
        return optimizer, lr_scheduler

    @staticmethod
    def create_rwkv_model(load_model, n_embd=-1, n_layer=-1, vocab_size=-1, **kwargs):
        """
        创建并初始化RWKV模型实例
        
        参数:
            load_model: 模型路径或预训练模型名称
            n_embd: 嵌入维度，-1表示使用默认值
            n_layer: 层数，-1表示使用默认值
            vocab_size: 词汇表大小，-1表示使用默认值
            **kwargs: 其他传递给RWKV模型的参数
            
        返回:
            model: 初始化好的RWKV模型实例
        """
        # 在创建模型前检查CUDA可用性
        if RWKV.check_cuda_available():
            logger.info("CUDA可用，将使用GPU创建模型")
            
            # 检查GPU内存状态
            allocated, reserved, total = RWKV.get_gpu_memory_usage()
            logger.info(f"当前GPU内存使用: {allocated:.2f}GB / {total:.2f}GB (使用率: {allocated/total:.2f})")
            
            # 如果内存使用率过高，尝试清理
            if allocated / total > MEM_MONITOR_THRESHOLD * 0.7:
                logger.info("GPU内存使用率较高，尝试清理...")
                gc.collect()
                torch.cuda.empty_cache()
        else:
            logger.warning("CUDA不可用，将在CPU上创建模型，性能可能受到影响")
            
        # 创建模型实例
        model = RWKV_Dual(
            n_embd=n_embd,
            n_layer=n_layer,
            vocab_size=vocab_size,
            load_model=load_model,
            **kwargs
        )
        
        return model
        
    @staticmethod
    def optimize_model_memory(model):
        """
        优化模型内存使用
        
        参数:
            model: RWKV模型实例
            
        返回:
            model: 优化后的模型实例
        """
        # 确保模型在评估模式下
        model.eval()
        
        # 清理不必要的梯度
        for param in model.parameters():
            param.requires_grad_(False)
        
        # 清理GPU内存
        if hasattr(model, 'clear_gpu_memory'):
            model.clear_gpu_memory(force=True)
        else:
            gc.collect()
            torch.cuda.empty_cache()
            
        return model

    @staticmethod
    def get_model_size(model):
        """
        获取模型大小信息
        
        参数:
            model: RWKV模型实例
            
        返回:
            size_info: 包含模型大小信息的字典
        """
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算模型内存占用（粗略计算）
        param_size_mb = total_params * 4 / (1024 * 1024)  # 假设FP32，每个参数4字节
        
        # 获取模型结构信息
        n_layer = getattr(model, 'n_layer', -1)
        n_embd = getattr(model, 'n_embd', -1)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_size_mb': param_size_mb,
            'n_layer': n_layer,
            'n_embd': n_embd
        }
    
    @staticmethod
    def apply_gradient_checkpointing(model, enable=True):
        """
        应用梯度检查点以减少内存使用
        
        参数:
            model: RWKV模型实例
            enable: 是否启用梯度检查点
            
        返回:
            model: 配置后的模型实例
        """
        if hasattr(model, 'gradient_checkpointing_enable') and enable:
            model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点以减少内存使用")
        elif hasattr(model, 'gradient_checkpointing_disable') and not enable:
            model.gradient_checkpointing_disable()
            logger.info("已禁用梯度检查点")
        else:
            logger.warning("模型不支持梯度检查点功能")
        
        return model
    
    @staticmethod
    def create_cosine_lr_scheduler(optimizer, max_steps, warmup_steps=0, min_lr_ratio=0.1):
        """
        创建余弦学习率调度器
        
        参数:
            optimizer: 优化器实例
            max_steps: 最大训练步数
            warmup_steps: 预热步数
            min_lr_ratio: 最小学习率与初始学习率的比例
            
        返回:
            lr_scheduler: 学习率调度器
        """
        from transformers import get_cosine_schedule_with_warmup
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            num_cycles=0.5,
            min_lr_ratio=min_lr_ratio
        )
        
        return lr_scheduler
    
    @staticmethod
    def monitor_memory_usage():
        """
        监控当前GPU内存使用情况
        
        返回:
            memory_info: 包含内存使用信息的字典
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA不可用'}
        
        # 获取当前设备
        device = torch.cuda.current_device()
        
        # 获取内存信息
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        
        # 获取总内存
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
        
        # 计算使用率
        utilization = allocated / total
        
        # 检查是否需要警告
        warning = utilization > MEM_MONITOR_THRESHOLD
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'total_gb': total,
            'utilization': utilization,
            'warning': warning
        }
