"""Parent-process RSS monitor remains responsive while CRFsuite holds the GIL."""
import subprocess
import time
import psutil
from src.utils.io import write_json


def limits():
    memory = psutil.virtual_memory()
    # At most half physical memory, and 60% of currently available memory.
    threshold = min(memory.total * 0.5, memory.available * 0.6)
    return {"memory_total_bytes": memory.total, "memory_available_bytes": memory.available,
            "maximum_memory_bytes": int(threshold), "minimum_available_bytes": int(memory.total * 0.1),
            "reason": "Budget=min(50% physical RAM,60% available RAM); stop at 90% budget or below 10% physical RAM available"}


def exceeded(rss, available, policy, fraction=0.9):
    return rss >= policy["maximum_memory_bytes"] * fraction or available < policy["minimum_available_bytes"]


def monitor(command, log_path, output_path, policy, poll=0.25):
    peak, started, stopped = 0, time.perf_counter(), False
    with open(log_path, "w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    processes = [process] + process.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak = max(peak, rss)
                    if exceeded(rss, psutil.virtual_memory().available, policy):
                        stopped = True
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(poll)
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    result = dict(policy, peak_rss_bytes_sampled=peak, polling_seconds=poll,
                  elapsed_seconds=time.perf_counter() - started, memory_limit_interruption=stopped, exit_code=code)
    write_json(output_path, result)
    if stopped or code:
        raise RuntimeError(f"Worker stopped: memory_limit={stopped}, exit_code={code}; see {log_path}")
    return result
