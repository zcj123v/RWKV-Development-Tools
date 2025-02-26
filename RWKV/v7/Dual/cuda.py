import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import logging
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.cpp_extension import load
import types
from RWKV.v6.state import BlockStateList
from typing import Union, Optional, List, Tuple
from config import global_config
HEAD_SIZE = global_config.train_service_config.model.head_size

# 配置日志记录器
logger = logging.getLogger(__name__)

# 内存监控阈值，当GPU内存使用率超过此值时触发清理
MEM_MONITOR_THRESHOLD = 0.8  # 默认为80%

def __nop(ob):
    """空操作函数，用于JIT编译开关"""
    return ob

MyModule = nn.Module
MyFunction = __nop
# if os.environ["RWKV_JIT_ON"] == "1":
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method

CHUNK_LEN = 24

full_parent_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CUDA编译标志
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

# 检查CUDA是否可用
CUDA_AVAILABLE = torch.cuda.is_available()

# 尝试加载CUDA扩展
try:
    if CUDA_AVAILABLE:
        load(name="wind_backstepping", 
             sources=[f'{full_parent_dir}/cuda/wkv7_cuda.cu', 
                      f'{full_parent_dir}/cuda/wkv7_op.cpp'], 
             is_python_module=False, 
             verbose=True, 
             extra_cuda_cflags=flags)
        logger.info("RWKV CUDA核心已成功加载")
        CUDA_LOADED = True
    else:
        logger.warning("CUDA不可用，将使用CPU实现")
        CUDA_LOADED = False
except Exception as e:
    logger.error(f"加载CUDA核心时出错: {str(e)}")
    logger.warning("将使用CPU实现作为备选")
    CUDA_LOADED = False

class WindBackstepping(torch.autograd.Function):
    """RWKV v7 的核心计算函数，使用CUDA实现高效的前向和反向传播"""
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape 
        assert T % CHUNK_LEN == 0, f"序列长度 {T} 必须是 CHUNK_LEN ({CHUNK_LEN}) 的倍数"
        
        # 检查数据类型
        if not all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b]):
            logger.warning("输入张量不是 bfloat16 类型，正在转换...")
            w, q, k, v, z, b = [i.to(dtype=torch.bfloat16) for i in [w, q, k, v, z, b]]
        
        # 确保连续性
        if not all(i.is_contiguous() for i in [w, q, k, v, z, b]):
            logger.warning("输入张量不是连续的，正在转换...")
            w, q, k, v, z, b = [i.contiguous() for i in [w, q, k, v, z, b]]
        
        # 分配输出和中间状态张量
        y = torch.empty_like(v)
        s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        
        # 调用CUDA核心
        try:
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        except Exception as e:
            logger.error(f"CUDA前向传播失败: {str(e)}")
            raise RuntimeError(f"CUDA前向传播失败: {str(e)}")
        
        # 保存上下文用于反向传播
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        # 获取保存的张量
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        
        # 检查梯度张量
        if dy.dtype != torch.bfloat16:
            logger.warning("梯度张量不是 bfloat16 类型，正在转换...")
            dy = dy.to(dtype=torch.bfloat16)
        
        if not dy.is_contiguous():
            logger.warning("梯度张量不是连续的，正在转换...")
            dy = dy.contiguous()
        
        # 分配梯度张量
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        
        # 调用CUDA核心的反向传播
        try:
            torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        except Exception as e:
            logger.error(f"CUDA反向传播失败: {str(e)}")
            raise RuntimeError(f"CUDA反向传播失败: {str(e)}")
        
        return dw, dq, dk, dv, dz, db

def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    """
    RWKV v7 的CUDA核心计算函数包装器
    
    参数:
        q, w, k, v, a, b: 输入张量
        
    返回:
        输出张量
    """
    # 检查CUDA是否已加载
    if not CUDA_LOADED:
        return _run_cpu_fallback(q, w, k, v, a, b)
    
    # 获取张量形状
    B, T, HC = q.shape
    
    # 检查输入
    if HC % 64 != 0:
        raise ValueError(f"隐藏维度 {HC} 必须是64的倍数")
    
    # 检查设备
    if not all(t.device.type == 'cuda' for t in [q, w, k, v, a, b]):
        logger.warning("输入张量不在CUDA设备上，正在转移...")
        device = torch.device('cuda')
        q, w, k, v, a, b = [t.to(device) for t in [q, w, k, v, a, b]]
    
    # 检查数据类型
    if not all(t.dtype == torch.bfloat16 for t in [q, w, k, v, a, b]):
        logger.warning("输入张量不是 bfloat16 类型，正在转换...")
        q, w, k, v, a, b = [t.to(dtype=torch.bfloat16) for t in [q, w, k, v, a, b]]
    
    # 确保连续性
    if not all(t.is_contiguous() for t in [q, w, k, v, a, b]):
        logger.warning("输入张量不是连续的，正在转换...")
        q, w, k, v, a, b = [t.contiguous() for t in [q, w, k, v, a, b]]
    
    # 检查序列长度是否为CHUNK_LEN的倍数
    if T % CHUNK_LEN != 0:
        # 填充到CHUNK_LEN的倍数
        pad_len = CHUNK_LEN - (T % CHUNK_LEN)
        logger.warning(f"序列长度 {T} 不是 CHUNK_LEN ({CHUNK_LEN}) 的倍数，正在填充 {pad_len} 个时间步...")
        
        # 创建填充张量
        pad_shape = (B, pad_len, HC)
        q_pad = torch.zeros(pad_shape, dtype=q.dtype, device=q.device)
        w_pad = torch.zeros(pad_shape, dtype=w.dtype, device=w.device)
        k_pad = torch.zeros(pad_shape, dtype=k.dtype, device=k.device)
        v_pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        a_pad = torch.zeros(pad_shape, dtype=a.dtype, device=a.device)
        b_pad = torch.zeros(pad_shape, dtype=b.dtype, device=b.device)
        
        # 连接填充张量
        q = torch.cat([q, q_pad], dim=1)
        w = torch.cat([w, w_pad], dim=1)
        k = torch.cat([k, k_pad], dim=1)
        v = torch.cat([v, v_pad], dim=1)
        a = torch.cat([a, a_pad], dim=1)
        b = torch.cat([b, b_pad], dim=1)
        
        # 更新T
        new_T = T + pad_len
        
        # 调用CUDA核心
        result = _run_cuda_core(q, w, k, v, a, b)
        
        # 移除填充部分
        return result[:, :T, :]
    else:
        # 直接调用CUDA核心
        return _run_cuda_core(q, w, k, v, a, b)

def _run_cuda_core(q, w, k, v, a, b):
    """
    调用CUDA核心的内部函数
    
    参数:
        q, w, k, v, a, b: 输入张量
        
    返回:
        输出张量
    """
    B, T, HC = q.shape
    
    # 重塑张量以匹配CUDA核心的输入格式
    q_reshaped = q.view(B, T, HC//64, 64)
    w_reshaped = w.view(B, T, HC//64, 64)
    k_reshaped = k.view(B, T, HC//64, 64)
    v_reshaped = v.view(B, T, HC//64, 64)
    a_reshaped = a.view(B, T, HC//64, 64)
    b_reshaped = b.view(B, T, HC//64, 64)
    
    # 调用CUDA核心
    try:
        result = WindBackstepping.apply(w_reshaped, q_reshaped, k_reshaped, v_reshaped, a_reshaped, b_reshaped)
        return result.view(B, T, HC)
    except Exception as e:
        logger.error(f"CUDA核心执行失败: {str(e)}")
        logger.warning("回退到CPU实现...")
        return _run_cpu_fallback(q, w, k, v, a, b)

def _run_cpu_fallback(q, w, k, v, a, b):
    """
    当CUDA不可用或执行失败时的CPU备选实现
    
    参数:
        q, w, k, v, a, b: 输入张量
        
    返回:
        输出张量
    """
    logger.warning("使用CPU备选实现，性能将显著降低")
    
    B, T, HC = q.shape
    H = HC // 64
    N = 64
    
    # 重塑为更易于处理的形状
    q = q.view(B, T, H, N)
    w = w.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    a = a.view(B, T, H, N)
    b = b.view(B, T, H, N)
    
    # 初始化输出
    y = torch.zeros_like(v)
    
    # 对每个头分别处理
    for h in range(H):
        # 初始化状态
        s = torch.zeros(B, N, N, device=q.device, dtype=torch.float32)
        
        # 按时间步处理
        for t in range(T):
            # 计算当前时间步的状态更新
            wt = torch.exp(-torch.exp(w[:, t, h]))
            kt = k[:, t, h]
            vt = v[:, t, h]
            at = a[:, t, h]
            bt = b[:, t, h]
            
            # 更新状态
            vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)
            ab = at.unsqueeze(-1) @ bt.unsqueeze(-2)
            s = s * wt.unsqueeze(-1).unsqueeze(-1) + s @ ab + vk
            
            # 计算输出
            y[:, t, h] = (s @ q[:, t, h].unsqueeze(-1)).squeeze(-1)
    
    # 重塑回原始形状
    return y.view(B, T, HC)

def check_cuda_availability():
    """
    检查CUDA是否可用，并返回详细信息
    
    返回:
        dict: 包含CUDA可用性和设备信息的字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_loaded": CUDA_LOADED,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory": props.total_memory / (1024**3),  # GB
                "compute_capability": f"{props.major}.{props.minor}"
            })
    
    return info

def monitor_gpu_memory(threshold=MEM_MONITOR_THRESHOLD, force_gc=False):
    """
    监控GPU内存使用情况，并在超过阈值时清理
    
    参数:
        threshold: 触发清理的内存使用率阈值
        force_gc: 是否强制执行垃圾回收
        
    返回:
        dict: 包含内存使用情况的字典
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    # 获取当前设备
    device = torch.cuda.current_device()
    
    # 获取内存信息
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024**3)    # GB
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
    
    # 计算使用率
    usage_ratio = allocated / total
    
    # 如果超过阈值或强制清理，执行垃圾回收
    if force_gc or usage_ratio > threshold:
        logger.info(f"执行GPU内存清理 (使用率: {usage_ratio:.2f}, 阈值: {threshold:.2f})")
        gc.collect()
        torch.cuda.empty_cache()
        
        # 重新获取内存信息
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        usage_ratio = allocated / total
    
    return {
        "available": True,
        "device": device,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "usage_ratio": usage_ratio,
        "threshold": threshold,
        "cleaned": force_gc or usage_ratio > threshold
    }

def optimize_cuda_performance():
    """
    优化CUDA性能的设置
    """
    if not torch.cuda.is_available():
        return
    
    # 设置CUDA性能优化选项
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 设置CUDA分配器选项以减少内存碎片
    torch.cuda.set_per_process_memory_fraction(0.9)  # 限制使用90%的GPU内存
    
    logger.info("已应用CUDA性能优化设置")

def clear_gpu_memory(force=False):
    """
    清理GPU内存
    
    参数:
        force: 是否强制清理
    """
    return monitor_gpu_memory(force_gc=force)

# 初始化时优化CUDA性能
if CUDA_AVAILABLE:
    optimize_cuda_performance()
    logger.info(f"CUDA信息: {check_cuda_availability()}")

