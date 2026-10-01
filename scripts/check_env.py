#!/usr/bin/env python3
"""环境自检：逐个 import 依赖、打印版本与平台信息。

只检查，不安装。缺包时退出码为 1 并给出建议命令。
在 openmm_dev 环境中运行：source env/activate.sh && python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import sys

# 模块名 -> (pip 安装建议, mamba 安装建议)
SUGGESTIONS = {
    "numpy": ("pip install numpy", "mamba install -n openmm_dev -c conda-forge numpy"),
    "scipy": ("pip install scipy", "mamba install -n openmm_dev -c conda-forge scipy"),
    "torch": ("pip install torch", "mamba install -n openmm_dev -c pytorch torch"),
    "openmm": ("pip install openmm", "mamba install -n openmm_dev -c conda-forge openmm"),
    "pytest": ("pip install pytest", "mamba install -n openmm_dev -c conda-forge pytest"),
    "matplotlib": ("pip install matplotlib", "mamba install -n openmm_dev -c conda-forge matplotlib"),
    "PeptideBuilder": ("pip install PeptideBuilder", "pip install PeptideBuilder"),
    "Bio": ("pip install biopython", "pip install biopython"),
}

REQUIRED = list(SUGGESTIONS)


def main() -> int:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 — 任何导入失败都算缺失
            missing.append(mod)
            print(f"[MISS] {mod}: {exc}")

    for mod, ver in versions.items():
        print(f"[ OK ] {mod} {ver}")

    if "openmm" in versions:
        import openmm as mm

        platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        print(f"openmm platforms: {platforms}")
    else:
        platforms = []

    cuda = False
    if "torch" in versions:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda}")

    if missing:
        print("\n缺失依赖：", ", ".join(missing))
        print("建议命令（脚本不会自动安装）：")
        for mod in missing:
            print(f"  {SUGGESTIONS[mod][0]}")
            print(f"  {SUGGESTIONS[mod][1]}")
        return 1

    ok = "CUDA" in platforms and cuda
    print(f"\n环境{'完整' if ok else '完整（无 CUDA，仅 CPU 可用）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
