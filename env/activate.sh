#!/bin/bash
# TANGRAM 环境激活：openmm_dev (mamba) + PYTHONPATH 指向仓库根目录
# 用法：source env/activate.sh（bash/zsh 均可）
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

# bash 用 BASH_SOURCE；zsh source 时该变量为空，回退 $0（zsh 会把它设为被 source 的文件）
_SRC="${BASH_SOURCE[0]:-$0}"
_SOCG_ROOT="$(cd "$(dirname "$_SRC")/.." && pwd)"
if [ ! -f "${_SOCG_ROOT}/socg/__init__.py" ]; then
  _SOCG_ROOT="$(pwd)"    # 兜底：路径解析失败时假定在仓库根目录
fi
case ":${PYTHONPATH:-}:" in
  *":${_SOCG_ROOT}:"*) ;;
  *) export PYTHONPATH="${_SOCG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
echo "[env] openmm_dev activated, PYTHONPATH=${PYTHONPATH}"
