# MiniCPM5-1B Camera Agent：300 条 SFT MVP

这个仓库先把 **数据 → 验证 → MiniCPM5-1B BF16 基线 → LoRA 试训 → 同口径评测 → 合并验证** 跑通。训练在你的云端 A6000 服务器手动执行。

**已交付本地可验证部分：**现有 10 工具契约、冻结 50 条回归题、300 条候选、生成/验证/清单工具、真实 tokenizer 的逐轮 mask 检查。**尚未执行：**云端 BF16 推理、GPU 训练、独立语义审核、真实摄像头闭环。不能把本地校验称为这些阶段已通过。

**本轮采用离线链路试训：`data/mvp-reviewed-v1`。** 你尚未人工审核，因此不要求不存在的 `human-reviewed-v1.jsonl`。使用包内同作者 assistant 复核记录：`data/reviews/author-v1/results/reviews.jsonl`，prepare/train 显式加 `--smoke`。300 条仍未通过人工/独立审核，不能据本次试训作正式效果验收。

- [MVP 方案与数据配比](docs/MVP.md) · [HTML 阅读版](docs/MVP.html)
- [让 agent 生成和审核数据](docs/AGENT_GENERATION.md)
- [本地验证记录](docs/VALIDATION.md)
- [生成器、验证器与本次审核说明](docs/DATA_REVIEW.md) · [逐条 HTML 审核记录](docs/数据生成与验证：300条审核记录.html)
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

## 1. 上传并校验完整离线包

本次交付 `minicpm5-sft-offline-20261022.tar.gz` 和同名 `.sha256`：包含源码、300 条数据及切分/种子/审核记录、冻结 50 题、完整官方 BF16 权重与 tokenizer。模型 revision 固定为 `87179e5c1f455ef22e6223592d2d61351b525bfc`；权重约 2.16 GB，不是 LoRA 或量化权重。包内没有 venv 或依赖 wheel，复用你已配置好的服务器环境。

在服务器把两个文件放到同一上传目录。保持现有 venv 已激活，在该目录逐条执行；报错即停止。解压到新的空目录，保留此前下载失败的 `models/base` 和旧实验记录：

```bash
sha256sum -c minicpm5-sft-offline-20261022.tar.gz.sha256
mkdir minicpm5-offline-run-20261022
tar -xzf minicpm5-sft-offline-20261022.tar.gz -C minicpm5-offline-run-20261022
cd minicpm5-offline-run-20261022/minicpm5-sft
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
python scripts/offline_bundle.py verify --root .
python -c "import sys, camera_sft; print(sys.executable); print(camera_sft.__file__)"
```

核对 Python 来自已有 venv，`camera_sft.__file__` 来自本次解压目录。若尚未激活环境，先执行 `source /你原项目的绝对路径/.venv/bin/activate`，替换实际路径。新开终端时重新激活并设置三个离线变量。**后续所有命令在新解压的仓库根目录执行，不执行 `cloud fetch`。**

`SOURCE_COMMIT` 标记源码版本，`OFFLINE_MANIFEST.json` 绑定逐文件大小与 SHA256，`MODEL_PROVENANCE.json` 记录上游权重哈希。校验用于检查传输完整性，不代表 GPU 已可运行。GitHub 仅保存代码和文档；模型和完整离线包不提交到 Git。

## 2. 复用已经配置好的 venv

你已经完成环境配置，**不用重建 venv、不用再次下载依赖**。先做检查：

```bash
python -m pip check
python -c "import torch; assert torch.__version__.split('+')[0] == '2.7.1'; assert torch.version.cuda == '12.6'; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
mkdir -p runs/environment
python -m pip freeze > runs/environment/pip-freeze.txt
nvidia-smi > runs/environment/nvidia-smi.txt
```

<details><summary>仅在另建机器、尚未配置环境时查看原 venv 安装步骤（需要访问依赖源）</summary>

下面按 Linux、Python 3.11、NVIDIA 驱动兼容 CUDA 12.6 的服务器写。先用 `nvidia-smi` 确认实际 GPU、空闲显存和驱动。若驱动不兼容，先由服务器管理员确定可用 PyTorch CUDA 组合，不改宿主驱动来迁就此命令。

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 10
python -m pip install -r requirements-train.txt
python -m pip install -e .
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
mkdir -p runs/environment
python -m pip freeze > runs/environment/pip-freeze.txt
```

直接依赖已固定；`requirements-train.txt` 不是完整的传递依赖锁。保存 `pip-freeze.txt`、驱动信息和每次实验 `run.json`，才能复现实际云端环境。这里的 GPU 环境尚未在你的服务器验证。

PyTorch 下载超时的阿里云备用源、SSH 代理和版本检查，见 [安装排障](docs/TORCH_INSTALL.md)。


</details>

## 3. 校验数据与包内 assistant 复核记录

```bash
python -m unittest discover -s tests -v
python -m camera_sft.data verify --data data/mvp-reviewed-v1
python -m camera_sft.data review --data data/mvp-reviewed-v1 \
  --reviews data/reviews/author-v1/results/reviews.jsonl \
  --allow-author-review --out runs/author-review-check.json
```

预期数据 `ok: true`、`train: 240`、`dev: 60`；作者复核完整性 `ok: true`，但 `independent_acceptance: false`，这是本次正常状态。不要创建空白人工文件、把作者签核改成独立签核或删除审核门槛。

之前的 `No such file: human-reviewed-v1.jsonl` 是因为还没有人工导出该文件。CAS Client / file reconstruction 错误发生在 Hugging Face 下载链路，仅凭摘要不能确定具体失败节点；本包已在本机取得完整文件，服务器无需访问 Hugging Face 或 CAS/CDN。

## 4. 先验证训练输入，暂不训练

```bash
python -m camera_sft.cloud prepare \
  --model-dir models/base --data data/mvp-reviewed-v1 \
  --reviews data/reviews/author-v1/results/reviews.jsonl \
  --out runs/prepared-mvp --smoke
```

预期得到 **396 个训练 assistant 步、104 个开发 assistant 步**；它们仍来自 240/60 个任务，不能把 500 个生成步骤说成 500 个独立样本。当前实际 tokenizer 检查的最长序列为 **1936 tokens**，全部低于 4096，未做截断。

检查 `runs/prepared-mvp/template-audit.jsonl`：每一步的推理前缀一致，XML、工具回填和 EOS 保留；prompt 和工具返回不参与 loss。直接使用官方模板，而非改写成简化 ChatML。本 MVP 使用 Transformers + PEFT 的小型训练器，目的是把这条协议检查直接接入训练；后续再迁移到 LLaMA-Factory 也必须保持同样的 token/mask 检查。

本轮 `--smoke` 允许作者复核数据用于链路试训，同时仍检查审核记录 ID、hash、accept 与理由。准备清单、训练 run.json 都保留 smoke 状态。不能把候选变成人工审核通过。

以后完成人工审核时，将 HTML 导出的文件放到 `data/reviews/human-reviewed-v1.jsonl`，先用不带 `--allow-author-review` 的 `data review` 校验；prepare 改用人工记录并去掉 `--smoke`，train 同样去掉它。使用新输出目录，保留这次 smoke 证据。

## 5. 保存 BF16 原版基线

使用空闲 GPU；下面 `0` 是示例编号。两个任务依次运行，不修改解码预算：

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --data data/mvp-reviewed-v1 --suite dev --out runs/base-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --data data/mvp-reviewed-v1 --suite frozen50 --out runs/base-50
```

每次保存逐条 `predictions.jsonl`、模型/数据身份、原始输出 tokens 和 `summary.json`。这里重新测的是 BF16；历史 FP32 的 32/50 仅是背景，不能拿来代替这次基线。

## 6. LoRA 试训，再用完全相同的输入评测

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud train \
  --model-dir models/base --prepared runs/prepared-mvp \
  --out runs/lora-mvp --smoke
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --adapter runs/lora-mvp/adapter \
  --data data/mvp-reviewed-v1 --suite dev --out runs/lora-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/base --adapter runs/lora-mvp/adapter \
  --data data/mvp-reviewed-v1 --suite frozen50 --out runs/lora-50
python -m camera_sft.compare --before runs/base-dev --after runs/lora-dev --out runs/compare-dev.json
python -m camera_sft.compare --before runs/base-50 --after runs/lora-50 --out runs/compare-50.json
```

默认配置在 `configs/base.json`：micro-batch 1、梯度累积 16、学习率 1e-4、2 epoch、seed 42。若只想先检验训练能否启动，可在 train 命令添加 `--max-steps 3`，使用独立输出目录；3 step 的成绩没有效果结论。MVP 不支持 `torchrun`，不要把八卡命令直接套上去。

## 7. 合并并检查交付模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud merge \
  --model-dir models/base --adapter runs/lora-mvp/adapter --out models/merged-mvp
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/merged-mvp --data data/mvp-reviewed-v1 --suite dev --out runs/merged-dev
CUDA_VISIBLE_DEVICES=0 python -m camera_sft.cloud evaluate \
  --model-dir models/merged-mvp --data data/mvp-reviewed-v1 --suite frozen50 --out runs/merged-50
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
