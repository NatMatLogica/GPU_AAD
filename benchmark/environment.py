#!/usr/bin/env python
"""
Hardware/software environment capture for benchmark reproducibility.

Captures CPU, GPU, OS, library versions, and git state.
Embedded in every benchmark result file.

Usage:
    python -m benchmark.environment
"""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class GPUDevice:
    """Single GPU device info."""
    index: int
    name: str
    memory_mb: int
    compute_capability: str
    driver_version: str


@dataclass
class BenchmarkEnvironment:
    """Full hardware/software environment snapshot."""
    # Software
    python_version: str = ""
    numpy_version: str = ""
    numba_version: str = ""
    cuda_toolkit_version: str = ""
    aadc_version: str = ""
    os_info: str = ""
    # CPU
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0
    # GPU
    gpu_devices: List[GPUDevice] = field(default_factory=list)
    # Reproducibility
    git_hash: str = ""
    cli_args: str = ""
    random_seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [
            "=== Benchmark Environment ===",
            f"OS:         {self.os_info}",
            f"Python:     {self.python_version}",
            f"NumPy:      {self.numpy_version}",
            f"Numba:      {self.numba_version}",
            f"CUDA:       {self.cuda_toolkit_version}",
            f"AADC:       {self.aadc_version}",
            f"CPU:        {self.cpu_model}",
            f"Cores:      {self.cpu_cores_physical} physical / {self.cpu_cores_logical} logical",
            f"CPU MHz:    {self.cpu_freq_mhz:.0f}",
        ]
        if self.gpu_devices:
            for g in self.gpu_devices:
                lines.append(
                    f"GPU {g.index}:     {g.name} ({g.memory_mb} MB, "
                    f"CC {g.compute_capability}, driver {g.driver_version})"
                )
        else:
            lines.append("GPU:        None detected")
        lines.append(f"Git hash:   {self.git_hash}")
        lines.append(f"CLI args:   {self.cli_args}")
        lines.append(f"Seed:       {self.random_seed}")
        return "\n".join(lines)


def _get_cpu_model() -> str:
    """Read CPU model from /proc/cpuinfo (Linux) or sysctl (macOS)."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_cpu_freq() -> float:
    """Get CPU frequency in MHz."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("cpu MHz"):
                        return float(line.split(":", 1)[1].strip())
        # Fallback: try lscpu
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "CPU max MHz" in line or "CPU MHz" in line:
                val = line.split(":", 1)[1].strip().replace(",", ".")
                try:
                    return float(val)
                except ValueError:
                    pass
    except Exception:
        pass
    return 0.0


def _get_cpu_cores() -> tuple:
    """Return (physical, logical) core counts."""
    logical = os.cpu_count() or 0
    physical = logical
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5
            )
            cores_per_socket = 0
            sockets = 0
            for line in result.stdout.splitlines():
                if "Core(s) per socket:" in line:
                    cores_per_socket = int(line.split(":", 1)[1].strip())
                elif "Socket(s):" in line:
                    sockets = int(line.split(":", 1)[1].strip())
            if cores_per_socket > 0 and sockets > 0:
                physical = cores_per_socket * sockets
    except Exception:
        pass
    return physical, logical


def _get_gpu_devices() -> List[GPUDevice]:
    """Detect GPU devices via numba.cuda or nvidia-smi."""
    devices = []
    try:
        from numba import cuda
        if not cuda.is_available():
            return devices
        for i in range(len(cuda.gpus)):
            with cuda.gpus[i]:
                dev = cuda.get_current_device()
                cc = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                # Get memory
                try:
                    ctx = cuda.current_context()
                    mem_free, mem_total = ctx.get_memory_info()
                    mem_mb = mem_total // (1024 * 1024)
                except Exception:
                    mem_mb = 0
                # Get driver version from nvidia-smi
                driver = _get_nvidia_driver()
                devices.append(GPUDevice(
                    index=i,
                    name=dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name),
                    memory_mb=mem_mb,
                    compute_capability=cc,
                    driver_version=driver,
                ))
    except Exception:
        # Fallback: parse nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    devices.append(GPUDevice(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_mb=int(float(parts[2])),
                        compute_capability="",
                        driver_version=parts[3],
                    ))
        except Exception:
            pass
    return devices


def _get_nvidia_driver() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def _get_cuda_toolkit_version() -> str:
    """Get CUDA toolkit version."""
    try:
        from numba import cuda
        runtime_ver = cuda.runtime.get_version()
        return f"{runtime_ver[0]}.{runtime_ver[1]}"
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
                parts = line.split("release")
                if len(parts) > 1:
                    return parts[1].strip().split(",")[0].strip()
    except Exception:
        pass
    return "unavailable"


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _get_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "installed (no version)")
    except ImportError:
        return "not installed"


def capture_environment(seed: int = 42) -> BenchmarkEnvironment:
    """Capture full benchmark environment. Call at start of every benchmark run."""
    physical, logical = _get_cpu_cores()
    return BenchmarkEnvironment(
        python_version=platform.python_version(),
        numpy_version=_get_version("numpy"),
        numba_version=_get_version("numba"),
        cuda_toolkit_version=_get_cuda_toolkit_version(),
        aadc_version=_get_version("aadc"),
        os_info=platform.platform(),
        cpu_model=_get_cpu_model(),
        cpu_cores_physical=physical,
        cpu_cores_logical=logical,
        cpu_freq_mhz=_get_cpu_freq(),
        gpu_devices=_get_gpu_devices(),
        git_hash=_get_git_hash(),
        cli_args=" ".join(sys.argv),
        random_seed=seed,
    )


if __name__ == "__main__":
    env = capture_environment()
    print(env.summary())
