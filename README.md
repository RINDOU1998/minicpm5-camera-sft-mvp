# MiniCPM5-1B Camera Agent：300 条 SFT MVP

这个仓库先把 **数据 → 验证 → MiniCPM5-1B BF16 基线 → LoRA 试训 → 同口径评测 → 合并验证** 跑通。训练在你的云端 A6000 服务器手动执行。

**已交付本地可验证部分：**现有 10 工具契约、冻结 50 条回归题、300 条候选、生成/验证/清单工具、真实 tokenizer 的逐轮 mask 检查。**尚未执行：**云端 BF16 推理、GPU 训练、独立语义审核、真实摄像头闭环。不能把本地校验称为这些阶段已通过。

- [MVP 方案与数据配比](docs/MVP.md) · [HTML 阅读版](docs/MVP.html)
- [让 agent 生成和审核数据](docs/AGENT_GENERATION.md)
- [本地验证记录](docs/VALIDATION.md)
- [前期完整调研](docs/research/MiniCPM5-1B%20Camera%20Agent微调：数据、训练与验收方案.html)（历史规划，当前执行范围以本 README 为准）

## 这次固定什么

| 项目 | 本轮设置 |
|---|---|
| 基线 | 官方 `openbmb/MiniCPM5-1B`，BF16，不思考；revision 见 `configs/base.json` |
| 工具 | 现有 Camera 数据集的 10 个工具，原始 system/schema 不改 |
| 新数据 | 300 条完整轨迹，240 train / 60 dev；75 个种子记录扩展，73 个语义分组 |
| 独立性 | 同种子全部改写、相同工具参数组合留在同一 split；60 条只是开发集，不是隐藏盲测 |
| 回归集 | 原 50 条原始文件按 SHA256 冻结，禁止用于训练/改写生成 |
| 训练 | 单卡 BF16 LoRA，rank 16，2 epoch，4K 上限，assistant-only loss |
| 工具执行 | 本轮使用明确标记的模拟返回，不连接真实设备 |

八卡 A6000 足以支持后续多组对照。300 条 MVP 先用其中一张卡，减少分布式变量。原先规划的全量对照、数据扩量和实板部署不属于这一轮执行步骤。

## 1. 把仓库放到云端

在服务器克隆公开仓库：

```bash
git clone https://github.com/RINDOU1998/minicpm5-camera-sft-mvp.git minicpm5-sft
cd minicpm5-sft
git log -1 --oneline
```

若服务器无法访问 GitHub，也可传输交付的 Git bundle，把第一条改成 `git clone /你上传的位置/minicpm5-sft-mvp.bundle minicpm5-sft`。不要把模型或 API 密钥提交进 Git。后续所有命令均在仓库根目录执行；任何命令报错，先处理错误，不继续后面的训练步骤。

## 2. 建立独立 Python 环境

下面按 Linux、Python 3.11、NVIDIA 驱动兼容 CUDA 12.6 的服务器写。先用 `nvidia-smi` 确认实际 GPU、空闲显存和驱动。若驱动不兼容，先由服务器管理员确定可用 PyTorch CUDA 组合，不改宿主驱动来迁就此命令。

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-train.txt
python -m pip install -e .
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
mkdir -p runs/environment
python -m pip freeze > runs/environment/pip-freeze.txt
```

直接依赖已固定；`requirements-train.txt` 不是完整的传递依赖锁。保存 `pip-freeze.txt`、驱动信息和每次实验 `run.json`，才能复现实际云端环境。这里的 GPU 环境尚未在你的服务器验证。

## 3. 验证数据并下载固定基线

```bash
python -m unittest discover -s tests -v
python -m camera_sft.data verify --data data/mvp
python -m camera_sft.cloud fetch --model-dir models/base
```

预期数据检查输出 `ok: true`、`train: 240`、`dev: 60`。下载固定 revision 的原始 HF 权重与 tokenizer，生成 `models/base/snapshot_manifest.json`；没有 LoRA、没有量化基座。下载失败时保留现场，换一个新的模型目录重试；不得把不完整目录当作成功快照。

## 4. 先验证训练输入，暂不训练

```bash
python -m camera_sft.cloud prepare \
  --model-dir models/base --data data/mvp \
  --out runs/prepared-mvp --smoke
```

预期得到 **396 个训练 assistant 步、104 个开发 assistant 步**；它们仍来自 240/60 个任务，不能把 500 个生成步骤说成 500 个独立样本。当前实际 tokenizer 检查的最长序列为 **1936 tokens**，全部低于 4096，未做截断。

检查 `runs/prepared-mvp/template-audit.jsonl`：每一步的推理前缀一致，XML、工具回填和 EOS 保留；prompt 和工具返回不参与 loss。直接使用官方模板，而非改写成简化 ChatML。本 MVP 使用 Transformers + PEFT 的小型训练器，目的是把这条协议检查直接接入训练；后续再迁移到 LLaMA-Factory 也必须保持同样的 token/mask 检查。

`--smoke` 明确表示：本轮允许尚未独立语义审核的候选用于**链路试训**。去掉该开关时，必须提供全部通过、与样本 hash 匹配的独立审核记录（见生成指南）。它不把候选提升为正式训练集。

## 5. 保存 BF16 原版基线

使用空闲 GPU；下面 `0` 是示例编号。两个任务依次运行，不修改解码预算：

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --suite dev --out runs/base-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --suite frozen50 --out runs/base-50
```

每次保存逐条 `predictions.jsonl`、模型/数据身份、原始输出 tokens 和 `summary.json`。这里重新测的是 BF16；历史 FP32 的 32/50 仅是背景，不能拿来代替这次基线。

## 6. LoRA 试训，再用完全相同的输入评测

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud train \
  --model-dir models/base --prepared runs/prepared-mvp \
  --out runs/lora-mvp --smoke
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --adapter runs/lora-mvp/adapter \
  --suite dev --out runs/lora-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --adapter runs/lora-mvp/adapter \
  --suite frozen50 --out runs/lora-50
python -m camera_sft.compare --before runs/base-dev --after runs/lora-dev --out runs/compare-dev.json
python -m camera_sft.compare --before runs/base-50 --after runs/lora-50 --out runs/compare-50.json
```

默认配置在 `configs/base.json`：micro-batch 1、梯度累积 16、学习率 1e-4、2 epoch、seed 42。若只想先检验训练能否启动，可在 train 命令添加 `--max-steps 3`，使用独立输出目录；3 step 的成绩没有效果结论。MVP 不支持 `torchrun`，不要把八卡命令直接套上去。

## 7. 合并并检查交付模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud merge \
  --model-dir models/base --adapter runs/lora-mvp/adapter --out models/merged-mvp
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/merged-mvp --suite dev --out runs/merged-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/merged-mvp --suite frozen50 --out runs/merged-50
python -m camera_sft.compare --before runs/lora-dev --after runs/merged-dev --out runs/compare-merge-dev.json
python -m camera_sft.compare --before runs/lora-50 --after runs/merged-50 --out runs/compare-merge-50.json
```

查看是否出现合并后新增失败，结合输出 tokens 审查差异。BF16 合并可能有数值差异，不能在没测时宣称逐 token 一致。输出是完整 HF 模型，**不是板端 OM**；本轮不转 OM、不部署相机。

## 结果怎么看

| 输出 | 含义 |
|---|---|
| `routing_pass` | 支持题工具+全部参数+单调用完全正确；不调用题无工具和格式错误；截断失败 |
| `simulated_exact_chain` | 开发题首轮正确，模拟返回后最终答案与种子参考精确一致；非调用题回答精确一致 |
| `gains / losses` | 同一条样本微调前后的改善/退步，不能只看净分 |
| `raw_first_output_changes` | 合并等对照中的首轮 token 差异，用于定位，不是单独质量指标 |

最终答案目前只做保守精确匹配，会把语义正确的改写判失败；它是可复查的 MVP 指标，**不是完整语义裁判**。所有原文保留给人工复核。不把模拟器成功写成真实摄像头执行成功。

这轮完成条件：数据清单和协议校验通过；BF16 基线、训练、adapter 评测、合并评测完整落盘；失败样本可以逐条追溯。**不要求 300 条训练就达到 48/50**。≥48/50、独立盲测、通用能力保留和真实工具闭环是后续正式验收目标。

所有命令拒绝覆盖已存在的实验输出目录。中断后保留该目录，用新名字重跑；不覆盖失败证据。`runs/`、`models/`、虚拟环境与凭据均被 Git 忽略。
