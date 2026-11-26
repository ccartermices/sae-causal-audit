"""确定性锁（进 T1 前的工程闸门）。

为什么必须锁
------------
T0 阶段发现：固定 seed 后同一脚本的结果仍会跳（覆盖率 0.728–0.763、
Jaccard 0.25–0.67）。根因是 cuBLAS 的 non-deterministic reduction 与
DataLoader/随机数流未统一。T1 的结论要进正文，读数必须可复算。

用法
----
    import det_lock
    det_lock.lock(0)          # 在任何 torch 计算之前调用

注意
----
`CUBLAS_WORKSPACE_CONFIG` 必须在 CUDA 初始化之前设置，故放在模块导入时执行。
`use_deterministic_algorithms(True)` 会让部分算子直接报错而不是静默变慢——
这是有意的：T1 里若某算子不支持确定性，我们希望它当场失败。
"""

from __future__ import annotations

import os
import random


# 必须在任何 CUDA 调用之前写入进程环境
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

_LOCKED = False


def lock(seed: int = 0, strict: bool = True) -> dict:
    """把 Python / NumPy / torch 的随机流和 cuDNN·cuBLAS 行为全部定死。

    返回实际生效的设置，供落盘留证。
    """
    global _LOCKED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 关闭 TF32：A100 之后 TF32 默认开启，会让同脚本跨卡结果不同
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    _LOCKED = True
    return dict(seed=seed, strict=strict,
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                deterministic_algorithms=bool(
                    torch.are_deterministic_algorithms_enabled()),
                device=("cuda" if torch.cuda.is_available() else "cpu"))


def assert_locked() -> None:
    if not _LOCKED:
        raise RuntimeError("det_lock.lock() 尚未调用；T1 的任何训练都必须先锁确定性")
