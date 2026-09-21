"""Resource and hardware monitor for PyTorch neural models on Apple Silicon / CPU."""
import os
import platform
import psutil
import torch
from src.utils.resource_monitor import limits, exceeded


def get_device(preferred: str = "mps", allow_cpu_fallback: bool = True) -> torch.device:
    """Select compute device (MPS or CPU) based on preferences and availability."""
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif allow_cpu_fallback:
            return torch.device("cpu")
        else:
            raise RuntimeError("MPS device requested but not available on this machine")
    elif preferred == "cpu":
        return torch.device("cpu")
    else:
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise ValueError(f"Unknown preferred device: {preferred}")


def system_diagnostics() -> dict:
    """Collect system hardware, OS, Python, and PyTorch details."""
    mem = psutil.virtual_memory()
    mps_avail = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    info = {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_avail,
        "total_ram_bytes": mem.total,
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_bytes": mem.available,
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
    }
    return info


class NeuralResourceTracker:
    """Track memory RSS and execution time during training epochs."""

    def __init__(self, policy: dict = None):
        self.process = psutil.Process(os.getpid())
        self.policy = policy or limits()
        self.peak_rss = self.process.memory_info().rss

    def sample(self) -> dict:
        """Sample current process memory usage and check against budget."""
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        avail = psutil.virtual_memory().available
        is_exceeded = exceeded(rss, avail, self.policy)
        return {
            "current_rss_bytes": rss,
            "peak_rss_bytes": self.peak_rss,
            "peak_rss_gb": round(self.peak_rss / (1024 ** 3), 4),
            "available_ram_bytes": avail,
            "limit_exceeded": is_exceeded,
        }
