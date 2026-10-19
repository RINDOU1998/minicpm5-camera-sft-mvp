# 用 agent 生产候选与独立审核

数据工具不调用任何付费教师 API，也不需要密钥。现有 300 条可直接复现；更丰富的候选可以由当前 agent、另一个人工启动的 agent 或外部教师生成，再用同一验证器入库。

## A. 复现当前版本

```bash
python scripts/author_seed_bank.py
python -m camera_sft.data build --out runs/data-rebuild
python -m camera_sft.data verify --data runs/data-rebuild
```

`author_seed_bank.py` 是本轮作者种子的来源；它会重写 seed_bank，通常无需运行。若改变种子，原数据发布的 source hash 会失效，必须生成新版本，不覆盖 `data/mvp`。

## B. 给 agent 准备分批计划

```bash
python -m camera_sft.data plan --out runs/agent-generation-01
```

目录里包含完整计划、工具契约和 15 个每批 20 条的文件。**不要把 `data/frozen/cases50.json` 交给生成 agent。** 生成者只看计划和工具契约；冻结回归集由验证器检查。

可以给生成 agent 下面这段指令：

> 读取 contract.json 和分配给你的 batch-XX.jsonl。每条只改写用户请求的表达，使其自然且不改变场景、工具、参数、否定范围和动作数量；保留 sample id、seed id、group、split、expected、模拟结果及最终参考答案。每个参数必须保留 parameter_evidence 对应的规范词片段，这让 MVP 验证器能逐项查证。对于 previous_tool_result/current_query 对象，只改 current_query，保留状态事实。输出相同条数的 JSONL，不加 Markdown。不要读取冻结评测题；不要自称通过了审核；无法保证语义时标记到独立 notes 文件并保留原候选。

这是**受控表达扩展模式**。如果希望 agent 创造全新语义、参数组合或结果，应先修改并审核 seed bank/lexicon，重新规划 group/split，再产生新的数据版本，不能通过改 expected 绕过验证。

验证单批并保存错误报告：

```bash
python -m camera_sft.data validate \
  --candidates runs/agent-generation-01/candidate-01.jsonl \
  --expected-count 20 --out runs/agent-generation-01/check-01.json
```

全部候选合为一个 300 行 JSONL 后，再运行全局验证和冻结。单批通过不能证明跨批 ID、分组和重复检查通过。

```bash
python -m camera_sft.data build \
  --candidates runs/agent-generation-01/candidates-300.jsonl \
  --out runs/data-agent-01
```

发生验证错误时命令非零退出；根据 `id` 定位修正。不要删除失败样本再用不同分母称为 300 条完成。

## C. 独立语义审核

把发布目录的 `review.jsonl` 复制到新文件，例如 `runs/reviews-01.jsonl`。由未参与该批生成的审核者逐条填写，原发布目录不改动。

审核者得到契约、完整候选和验证告警，但不应看到生成者的自我辩护。对每条检查：

1. 是否真的应该调用；缺参、多动作、无需工具、不支持是否区分正确。
2. 所选工具及所有参数能否从用户/允许的历史中确定；规范词片段出现不等于语义必然正确。
3. 工具结果只是合成事实，最终回复是否与事实一致，是否把失败说成成功。
4. 是否为原评测题或其他 split 的近似改写；有疑问应 reject 并注明原因。

通过时填写 `verdict: "accept"`、真实 `reviewer`、`independent: true`、`notes`；不通过写 `reject`。`sample_sha256` 必须保留。工具只能检查记录是否齐全、hash 是否对应，无法证明审核者身份真实独立。

```bash
python -m camera_sft.data review \
  --data data/mvp --reviews runs/reviews-01.jsonl --out runs/review-check.json
python -m camera_sft.cloud prepare \
  --model-dir models/base --data data/mvp --reviews runs/reviews-01.jsonl \
  --out runs/prepared-reviewed
```

有 pending/reject 会非零退出，不能用于非 smoke 的 prepare。若修正数据，旧审核 hash 自动失效，必须重新审核相应版本。

## D. 清单文件

- `candidates.jsonl`：300 条母格式，含输入、标签、参数依据和来源。
- `train.jsonl / dev.jsonl`：固定分组切分，规则复核其内容与母格式一致。
- `validation.json`：覆盖、错误与相似性告警；明确语义审核状态。
- `review.jsonl`：逐条待审核记录，不伪造 accept。
- `manifest.json`：数据、review 模板、contract、seed、基线配置及原 50 条的 SHA256。

训练只使用 messages 经官方模板转换后的 token；expected、审核结果、分组和 gold 不进入模型输入。
