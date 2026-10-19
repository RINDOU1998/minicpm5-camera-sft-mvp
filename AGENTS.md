# MiniCPM5 Camera SFT MVP

本目录是后续数据、训练代码和文档的源码权威。用户已指定八卡 A6000、官方 MiniCPM5-1B BF16 基线、现有数据集的 10 个工具。目标为冻结 50 条 ≥48/50，加独立测试与真实工具闭环；300 条只验链路。

- 先检查 git status，保留已有修改。不要访问开发板或自动执行云端训练。
- `contracts/camera_v1.json` 保留冻结工具/schema/system 原文。`data/frozen/cases50.json` 只用于回归，禁止训练或生成其改写。
- 300 条按 group 切分 240 train / 60 dev；同一种子衍生数据不可跨 split。候选、规则通过、独立语义审核、GPU 训练通过、真实工具验收必须分别标记。
- 模拟工具结果必须显式标为 simulated；不能把模拟通过写成真实硬件成功。
- 通用问答仍携带当前工具和 system；对拒识分别区分正常回答、不支持、缺参、多动作。
- 不改变原词表、template 或工具协议来凑分。固定 HF revision，保留 assistant-only label mask 和渲染证据。
- 本地数据工具仅依赖 Python 标准库；训练依赖只装在独立云端虚拟环境。`python -m unittest discover -s tests -v` 验证失败路径。
- 提交前 `git diff --check`，显式暂存并检查索引。模型、runs、token、凭据不提交。推送仅到用户指定的远程，不自行公开数据。
- README 是云端手动执行的唯一入口；MVP 范围见 docs/MVP.md。历史研究位于 docs/research，不代表本次实验结果。
