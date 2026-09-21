"""Report installed versions and measured resources without requiring PyTorch."""
import argparse
import importlib.metadata
import importlib.util
import platform
import shutil
import sys
import psutil
from src.utils.io import now, write_json


def environment():
    memory = psutil.virtual_memory()
    result = {"timestamp": now(), "os": platform.platform(), "architecture": platform.machine(),
              "python": sys.version, "seed": 42, "memory_total_bytes": memory.total,
              "memory_available_bytes": memory.available, "disk_free_bytes": shutil.disk_usage(".").free,
              "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
              "torch_mps_available": "not measured: torch not installed"}
    if importlib.util.find_spec("torch"):
        import torch
        result["torch_mps_available"] = torch.backends.mps.is_available()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/environment/environment.json")
    args = parser.parse_args()
    write_json(args.output, environment())
