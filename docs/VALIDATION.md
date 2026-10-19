# 本轮本地验证与云端待验

日期：2026-10-19。只执行本地源码、数据与tokenizer检查；未连接开发板或云端服务器，未运行真实模型推理/微调。

| 检查 | 结果 |
|---|---|
| 工具/schema/system冻结 | 10工具、50题；原始文件逐字节保存并绑定SHA256 |
| 数据验证 | 300条；240train/60dev；75种子记录、73语义分组；规则错误0 |
| 泄漏筛查 | 无规范化精确重合；当前0条达到词面告警阈值；不证明没有语义泄漏 |
| 独立语义审核 | 待做，300条均未伪造审核通过 |
| 故障回归 | 18项Python unittest通过；覆盖协议拒绝、清单篡改、评测泄漏和不可比较的实验 |
| 官方tokenizer | 固定HF revision的真实tokenizer与chat_template；未下载1B权重 |
| 全量样本渲染与mask | 396训练步骤、104开发步骤；最长1936tokens；无截断；prefix与推理一致 |
| 原50条上下文预算 | 全部prompt+128输出预算可放入4K |
| 训练依赖API探针 | 随机微型Llama在CPU完成一步masked-loss训练、LoRA参数更新、adapter保存、合并和重载；不等于MiniCPM训练 |
| 依赖一致性 | `pip check`通过；本地torch为2.7.1+cpu，云端CUDA构建另验 |
| Linux 全新 Git 克隆 | Ubuntu-22.04 中 18 项测试、数据清单验证、数据重建逐字节比对通过；冻结证据按原字节保存，避免换行转换破坏 hash |
| 配对评测输入一致性 | 比较器检查消息、工具 hash、渲染 prompt hash 和 token IDs；仅数据文件相同不足以视为同条件 |
| 云端BF16/GPU训练 | 待用户手动执行，不以本地测试替代 |
| 真实工具闭环/OM | 本轮不执行 |

复核数据：`python -m camera_sft.data verify --data data/mvp`。故障用例：`python -m unittest discover -s tests -v`。

2026-10-19 审核后更新：`data/mvp-reviewed-v1` 保留 240/60，8 条设置查询已修正，75 个场景主体与 300 条实际前缀扩展完成同作者 assistant 复核；独立审核仍未通过。新增种子版本隔离、同作者审核不能通过独立门槛、拒绝/过期审核记录测试，当前共 **20 项**通过。修订数据的 tokenizer 准备摘要见 `docs/tokenizer-preflight-reviewed-v1.json`。旧数据和旧验证摘要保留，不混用。

人工审核 HTML 已用本机 Chrome + Playwright 检查：默认 300 条待审核、同组操作、退回理由必填、300 行 JSONL 导出/导入、过期 hash 拒绝且不破坏当前草稿、本地草稿恢复、390px 手机无横向溢出、无网络请求。1440px/390px 截图已目视检查。测试使用隔离浏览器与 `UI_TEST_ONLY` 身份，不写入真实人工审核文件；测试脚本在 `tests/review_ui.cjs`，依赖 Playwright，仅供可选本地 UI 检查，不是云端训练依赖。

原始本地运行输出保存在忽略的runs目录；可提交的tokenizer准备摘要在`docs/tokenizer-preflight.json`，微型模型探针结果在`docs/cpu-api-probe.json`。传递依赖/驱动/真实GPU内存尚需云端记录。输出目录不覆盖，失败不重标成功。
