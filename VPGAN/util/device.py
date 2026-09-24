"""本地适配（PathAI）：统一设备选择。

上游代码假设 CUDA 环境；本模块在无 CUDA 时自动回退
Apple Silicon MPS，再回退 CPU，供 options/base_model/clip_score 共用。
"""
import torch


def pick_device(gpu_ids=None):
    """gpu_ids 非空且有 CUDA -> cuda:i；否则 mps 可用 -> mps；否则 cpu。

    保持上游约定：`--gpu_ids -1` 解析后为空列表，表示强制 CPU。
    """
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
