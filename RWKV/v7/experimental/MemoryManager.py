# 内存管理工具类
import torch
import gc
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)

# 内存监控阈值，当GPU内存使用率超过此值时触发清理
MEM_MONITOR_THRESHOLD = 0.8  # 默认为80%

class MemoryManager:
    """内存和显存管理工具类"""
    
    @staticmethod
    def check_cuda_available():
        """检查CUDA是否可用"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU运行模型")
            return False
        return True
    
    @staticmethod
    def get_gpu_memory_usage():
        """获取当前GPU内存使用情况"""
        if not torch.cuda.is_available():
            return 0, 0
        
        # 获取当前设备
        device = torch.cuda.current_device()
        
        # 获取分配的内存
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        
        # 获取缓存的内存
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        
        # 获取总内存
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
        
        return allocated, reserved, total
    
    @staticmethod
    def clear_gpu_memory(force=False):
        """清理GPU内存"""
        if not torch.cuda.is_available():
            return
        
        allocated, reserved, total = MemoryManager.get_gpu_memory_usage()
        usage_ratio = allocated / total
        
        # 只有当内存使用率超过阈值或强制清理时才执行
        if force or usage_ratio > MEM_MONITOR_THRESHOLD:
            logger.info(f"清理GPU内存 (使用率: {usage_ratio:.2f})")
            gc.collect()
            torch.cuda.empty_cache()
            
            # 清理后再次检查
            new_allocated, _, _ = MemoryManager.get_gpu_memory_usage()
            logger.info(f"清理后GPU内存: {new_allocated:.2f}GB (减少了 {allocated - new_allocated:.2f}GB)")
    
    @staticmethod
    def optimize_tensor(tensor, dtype=None, device=None, inplace=False):
        """优化张量的内存使用"""
        if tensor is None:
            return None
            
        # 如果不是inplace操作，创建新张量
        if not inplace:
            if dtype is not None and device is not None:
                return tensor.to(dtype=dtype, device=device)
            elif dtype is not None:
                return tensor.to(dtype=dtype)
            elif device is not None:
                return tensor.to(device=device)
            return tensor
            
        # inplace操作
        if dtype is not None and device is not None:
            tensor.data = tensor.data.to(dtype=dtype, device=device)
        elif dtype is not None:
            tensor.data = tensor.data.to(dtype=dtype)
        elif device is not None:
            tensor.data = tensor.data.to(device=device)
        
        return tensor
