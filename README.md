# TANGRAM

> **TANGRAM** —— Stateful-Operator Coarse-Graining for Proteins
> （中文可称 **七巧**：少数几块固定形状的板，按通用规则拼出任意图形）

少数可复用的 **residue operator primitives**（每个残基 = bead + 姿态 + 氨基酸种类 +
离散内部态 *s*）与通用耦合规则，按"七巧板拼装"的方式构成多尺度动力学模型，
而不是为每条蛋白重新学习一个完整势能面。

动力学为连续坐标 Langevin（BAOAB）+ 离散内部态 Metropolis flip 的混合采样；
势函数对参数线性（RBF / Fourier 基 + 查表权重），由力匹配 + 变分似然联合拟合凸目标。

## 命名约定

| 场景 | 名字 |
|---|---|
| 论文 / 对外 / 文档标题 | **TANGRAM** |
| 中文 | 七巧 |
| 代码包名 / 目录 / import | `socg`（历史工作代号，保留不动） |

历史文档中出现的 **SOCG**（Stateful Operator Coarse Graining）即本项目的早期代号，
指向同一方法。

## 现状速览（2026-10-01）

- Phase 1（肽段 POC：Ala10 / CLN025）代码全部就位：`socg/` 包 + `scripts/` 10 个脚本。
- 快速层 pytest 78 passed / 4 skipped，慢速层 1 passed / 3 skipped（skip 均因缺 PeptideBuilder）。
- CG 管线已在 GPU 节点用合成数据端到端冒烟跑通。
- 环境激活：`source env/activate.sh`（conda 环境 `openmm_dev`）。

## 快速开始

```bash
git clone https://github.com/Cedrus810/TANGRAM
cd TANGRAM
source env/activate.sh          # mamba/conda 环境 openmm_dev + PYTHONPATH
pytest -q -m "not slow"         # 快速层测试（-m slow 为 GPU 慢速层）
```

AA 生产采样、拟合、CG 生产与 Gate 评估的完整命令序列见 `scripts/` 内各脚本的
`--help`（`run_aa.py` → `build_dataset.py` → `fit_model.py` → `run_cg.py` → `evaluate.py`）。

## License

[MIT](LICENSE)
