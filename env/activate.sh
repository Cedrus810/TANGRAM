#!/bin/bash
# SOCG 环境激活：openmm_dev (mamba) + PYTHONPATH 指向仓库根目录
# 用法：source env/activate.sh
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

_SOCG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case ":${PYTHONPATH:-}:" in
  *":${_SOCG_ROOT}:"*) ;;
  *) export PYTHONPATH="${_SOCG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
echo "[env] openmm_dev activated, PYTHONPATH=${PYTHONPATH}"
