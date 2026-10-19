# Camera-agent delivery evidence

This directory is a separate, **unqualified camera-finetune handoff**, not a stock
release replacement. The pending manifest template in this directory still
records all three configurations as `MISSING_CAMERA_OM` / `NOT_TESTED`: it has
not yet been populated with the external lab artifacts. This is template status,
not the current testing status: all three initial camera decodes exist; the
Euler Pi 8K/10K strict11 captures (2026-09-06) read 357/365 and 360/365 under
the v2 forced top-1 count and are cross-board reference only;
16K was interrupted by host network loss during G07. Release evidence is
ONE independently raw-audited target-board capture per context (gate
v3 below). See the opening README for measured results. No camera OM, checkpoint weights,
SDK libraries, board credentials or hardware-control permission is included.

`cpu_short_prompt_golden.json` preserves all 11 user prompts, their original
numbering (including both label 3 entries), complete rendered prompts/input IDs,
verbatim original HF CPU FP32 output text/token IDs and terminal EOS decisions.
It was extracted without new inference from the sealed source report identified
by SHA-256 inside the JSON. The portable subset intentionally omits local paths,
per-step logits and large binary references; the original report is not bundled.
The rendered prompts already contain the system/tool contract. Preserve it and
do not silently repair the model's XML or reinterpret legacy parameters.

These are **822–826-token, no-conversation-history inputs**, not 8K/10K/16K
long-history golden trajectories. Replaying them on a larger-context OM tests
that configuration's short-prompt behavior only. Long-history status remains
`NOT_TESTED` even if every short-prompt configuration later qualifies.
Matching the model's reference proves **fidelity to the HF golden, not
compliance**: it does not prove valid or safe camera actions. The 11 references
were generated under our 2026-09-04 tool contract, and 10 of them violate that
schema; the app's fail-closed filter rejects them by design (a runtime safety
property, not an OM defect). Whether they match the customer's training-time
tool contract is unknown — see the question below. Never dispatch this corpus.

## Lightweight validation

From the repository root, with Python 3.10 or newer (standard library only):

```sh
python3 release/camera_agent/validate_manifest.py --check-only
python3 release/camera_agent/validate_manifest.py
```

The first command accepts an honest pending or completed-but-unqualified
manifest. The second is the qualification gate and currently must exit nonzero.
Neither command runs inference. The validator binds artifact sizes/SHA-256,
re-reads them before returning, and recounts the reported per-step decisions; it
does **not** independently recompute full-vocabulary raw logits. A completed
comparison must also carry a hash-bound independent numeric-audit receipt. That
audit must be produced by an actual separate audit, never fabricated or treated
as a substitute for one.

## Release gate v3 (`forced_top1_agreement.v3`, one capture per context)

Manifest schema `camera-agent-release.v3`; comparison schema
`camera-agent-context-comparison.v3`; audit receipt schema
`camera-agent-context-audit.v3`. **Rule: ONE independently raw-audited capture
per context configuration.** The evidence for a context is a single complete
strict11 capture on the target board, raw-audited by the Orange-lane auditor
and projected to a portable comparison carrying the 365 per-step device top-1
decisions. At every one of the 365 golden generation steps (EOS decisions
included) the validator compares that decision with the golden
`generated_ids`, and the configuration qualifies when at least **362** agree.
The count is summed over the 11 cases per configuration, not judged per case.
Cosines and free-text / token / EOS / finish-reason exactness are validated
structurally and reported; they never decide qualification. The manifest's
per-context `forced_top1_agreement_count` must equal the comparison's own
claim and the validator's recount, `capture_board` / `capture_boot_id` must
equal the capture the comparison records, and `qualified` must follow the
count.

One capture, and what that costs (user ruling 2026-09-08, 「不用测2次，一次够了」):
the earlier K=3 consensus gate v2.1 is **replaced, not bypassed**. Shipping one
capture inside a schema that says "consensus of 3 independent captures" would
be false labelling, so the consensus vocabulary is gone from the shipped record
and v2.1 evidence is refused. What does not go away is the reason v2.1 existed:
the target board is **characterized non-deterministic**, so a single capture's
count is **one sample of a distribution**, not a reproducible number.

**Each context ships exactly ONE capture, and that capture's count is a single
sample of a device characterized as non-deterministic.** The same fixed-input
decode does not return byte-identical results on the Orange Pi Ai Fly board. The
Phase-1 characterization measured what varies and by how much: in the layer-6
attention Vector stage upstream of the softmax (`L6_scale` or `L6_mask_add`),
**every drifting element has index ≡ 15 (mod 16)** — the last lane of each
16-element group — and flips by **exactly ±1 code**, on all 16 rows
(2 channels × 8 heads), on byte-identical input. The Cube MatMul output feeding
that stage is byte-stable (153/153 exact pairs over 18 decodes); the softmax
output and the layer hidden state are not (53–54 distinct hidden patterns over
54 decodes). One ±1 code at a near-tie step can move that step's top-1, so
re-running the same capture on the same board with the same inputs may return a
different `forced_top1_agreement_count`. Those measurements are pinned in
`reports/orangepi-l6-repeatability-phase1-verdict-r1-20260908.json` (11185 B,
`23f172f9…`), `reports/orangepi-l6-phase1b-d1-r1-20260908.json` (28759 B,
`44b68bb4…`) and `ORANGEPI_LANE15_VECTOR_NONDETERMINISM_REPORT_20260908.md`
(67309 B, `d55dcbe7…`) — full digests in the known-issue table below.

Every v3 gate record therefore carries a `single_capture_sampling_caveat`
block — `evidence_captures_per_context: 1`, `device_repeatability:
"characterized non-deterministic"`, what varies, `count_is_one_sample: true`, a
reproducibility statement, and the three pinned characterization artifacts with
their sizes and SHA-256 — and the validator refuses a manifest that omits it or
edits a single character of it. Its `reproducibility` field reads, literally:

```text
forced_top1_agreement_count is ONE SAMPLE of a non-deterministic device, not a reproducible number: re-running the same capture on the same board with the same inputs may return a different count. Never quote it as reproducible.
```

The validator also repeats the whole block in its own verdict JSON, so the
count cannot be read anywhere without it. Never quote the count as reproducible.

### Evidence roster

Per context configuration the release ships exactly two evidence files, both
recorded in that context's manifest row as `{path, bytes, sha256}` records under
`test_evidence`:

| role | schema / status | what it holds |
|---|---|---|
| `comparison` | `camera-agent-context-comparison.v3` | the portable projection of the one target-board capture: identity fields, the 11 ordered cases with all 365 per-step device top-1 decisions, the `capture` block, and `forced_top1_agreement_count` |
| `independent_audit` | `camera-agent-context-audit.v3` / `PASS_INDEPENDENT_RAW_AUDIT` | the hash-bound receipt of the independent raw audit of that same capture: `comparison_sha256` plus a `capture` block equal to the comparison's, field for field |

Three contexts × one capture each = three comparisons and three receipts for the
whole release. Deliberately **not** shipped: the raw board-side capture data,
the camera OM, checkpoint weights, SDK libraries, board credentials, and the
three characterization artifacts behind the sampling caveat — those stay in the
gate working directory and are pinned here by size and digest, so the caveat
remains a claim a reader can go and check rather than a file we hand over.

v1, v2 and v2.1 manifests, comparisons and receipts are each refused by name
with no upgrade path: v1 carries no per-step device decisions; v2 pins no
capture board and carries no sampling caveat; v2.1's per-step values are a
2-of-3 mode of three captures (with `null` unstable steps), which is not one
capture's decisions. A v3 comparison must be projected from a target-board
capture by the v3 projector, which also writes the v3 receipt.

## Filling real evidence later

All file records are `{path, bytes, sha256}` relative to this directory. Missing
files, escaping paths, stock checkpoint identity, stale report bindings, unknown
fields and unsupported qualification claims fail closed. Do not add dummy OM
hashes or change the pinned golden to make a result pass.

For an available context, set `camera_om_status` to `AVAILABLE` and `camera_om` to
`{role: "decode", context_tokens, checkpoint_sha256, file}` using the actual
camera decode OM. It may remain `NOT_TESTED` with null `test_evidence`. After
real testing set `test_status` to `COMPLETED`; `test_evidence` contains file
records named `comparison` and `independent_audit`.

The comparison JSON schema is `camera-agent-context-comparison.v3`, with
exactly `schema`, `completed: true`, `context_tokens`, `checkpoint_sha256`,
`golden_sha256`, `decode_sha256`, `reference_scope`, `input_contract_sha256`,
`cases`, `capture` and `forced_top1_agreement_count`. `reference_scope` is
`short_prompt_camera11_on_context_configuration`, and
`forced_top1_agreement_count` is the projector's own count, which the validator
recounts from the cases and refuses on any drift. The `capture` block carries
exactly `board` (`machine_id`, `hostname`, `device_tree_model` — the stable
target-board identity, which is pinned in the validator), `boot_id` (recorded,
never pinned: the board reboots spontaneously), `board_report_sha256` (the
digest of that session's own board-side `report.json`), `board_output_root_sha256`
(the digest of the board-side capture output root it wrote into),
`bench_lock_tag` (the per-context bench lease tag, or `null`) and
`raw_audit_sha256` (the digest of the independent raw audit of this capture).
Everything but `board` is recorded rather than bound, so that a third party can
ask the payload preparer for *this* capture's raw data rather than for "a
capture".
The input-contract hash uses UTF-8 JSON, sorted keys, compact separators and
unescaped Unicode. All checkpoint and golden identities must match this handoff.

Each of the 11 ordered cases contains exactly `sequence`, `input_ids_sha256`
(compact JSON encoding of the complete integer input-ID array), `generated_text`,
`generated_ids` (including EOS when present), `finish_reason`, `eos_token_id`
(null on length exhaustion), `forced_device_top1_token_ids`, `forced_cosines`
and `free_aligned_cosines`.
`forced_device_top1_token_ids` is the capture's argmax decision at every golden
step while the golden prefix is teacher-forced — one entry per golden
generation step including the EOS decision, always an integer token id, never
`null`; one capture always decided, so there is no consensus mode and no
unstable step. It is what the gate counts. Cosines are raw full-vocabulary
comparisons against the same CPU full prefix, one per decision including EOS;
they are report-only. Forced arrays have the golden generation length.
Free arrays stop after the first differing generated token's decision: subsequent
input prefixes differ and must not be compared numerically across trajectories.

The audit receipt contains exactly `schema: "camera-agent-context-audit.v3"`,
`status: "PASS_INDEPENDENT_RAW_AUDIT"`, `comparison_sha256` (the comparison
file) and `capture` (equal, field for field, to the comparison's `capture`
block), so the receipt names *which* board capture was audited and not only
which file it read. It means that capture was independently raw-audited and
projected, not that accuracy passed. Preserve the actual comparison / auditor /
raw-data provenance with the testing delivery; this compact receipt alone is
not the underlying numeric evidence. Manifest context rows carry
`capture_board` and `capture_boot_id` (both null until completed), and they
must equal what the comparison's `capture` block records.

The gate computes each context's qualification rather than trusting a label:
the forced top-1 agreement count must be **at least 362 of 365**; 361 fails.
Completed evidence below the threshold is allowed but `qualified` must remain
false. Release `qualified` is true only when all three context configurations
pass; it never grants camera execution or long-history qualification.

## Euler Pi evidence is cross-board reference only

The 2026-09-06 Euler Pi 8K/10K captures, re-projected with the v2 auditor on
2026-09-08, read **357/365** and **360/365** forced top-1 agreements. They are
kept as cross-board reference evidence, never release inputs: the validator and
the payload preparer admit only capture chains whose board identity is the
target board (Orange Pi Ai Fly, machine_id `a3855d88…`) audited by the
target-board auditor, the comparison's `capture.board` must equal that pinned
identity together with the capture's boot id, and no v1, v2 or v2.1 portable
projection is ever admitted as release evidence.

## Known issue: target-board Vector non-determinism

The same fixed-input decode on the Orange Pi Ai Fly board does not return
byte-identical results. The Phase-1 characterization localizes the drift to
the layer-6 **Vector stage upstream of the softmax** — either `L6_scale`
(fused Clip / Mul(1/√128) / Clip, S16 → FP16 → S16) or `L6_mask_add` (the
vvadd against the attention mask); the tap that would separate those two was
not run. Its signature is that **every drifting element has index ≡ 15 (mod 16)**
— the last lane of each 16-element group — and flips by exactly **±1 code**,
with drift seen on all 16 rows (2 channels × 8 heads), on byte-identical
input. The Cube MatMul output that
feeds the stage (`L6_qk:0`) is byte-stable across the 18 decodes of the
scores-tap campaign (one cell, 6 processes × 3 decodes, 153/153 exact pairs),
and the softmax
itself is deterministic given its input, so the softmax output (`L6_probability`,
two fixed lane-15 sites) drifts only because its input already did. Cache mode,
process boundary, an inserted `bar.p8`, NPU clock and supply are ruled out with
evidence, and the same `.om` is byte-repeatable on the vendor SS928V100 DEMO
board (1448 raw files, 0 changed, two full strict11 captures on two days) — so
the runtime/SDK, not the OM, is the attributed source, and no Tier-0 byte edit
of ours can reach the mechanism.

Both artifacts live in the gate working directory and are **not** part of this
repository:

| artifact | bytes | sha256 |
|---|---:|---|
| `reports/orangepi-l6-repeatability-phase1-verdict-r1-20260908.json` (Phase-1 verdict: PV and masked taps, 54 decodes; records the Cube output as a candidate still to be discriminated) | 11185 | `23f172f921f00cdc4912fa1baa10ae1e28f1e6e577a18370427d09def7c40c51` |
| `reports/orangepi-l6-phase1b-d1-r1-20260908.json` (scores tap: the 18-decode campaign that made the Cube output byte-stable and excluded it) | 28759 | `44b68bb24620e6a65e11a713f0ce56a2e7da740bda9a7e16eed27ba756d51035` |
| `ORANGEPI_LANE15_VECTOR_NONDETERMINISM_REPORT_20260908.md` (vendor-facing report, written 2026-09-08) | 67309 | `d55dcbe7fa6e7afe98078cf995cf82b80c3468cd0e7d32b241678336d1ea70a8` |

Until the vendor or the community runtime maintainer resolves it, this is the
reason the v3 gate's `single_capture_sampling_caveat` exists: the release ships
ONE capture per context, and **that capture's `forced_top1_agreement_count` is
one sample of this non-deterministic device, not a reproducible number**.
Re-running the same capture on the same board with the same inputs may return a
different count. Do not quote it as reproducible, and do not present a passing
count as evidence that the board is deterministic.

## Published payload (2026-09-08)

The first v3 payload was published as a GitHub **prerelease** from this branch.
It is honestly **unqualified**: every configuration was captured once on the
target board, independently raw-audited, and every count is below the 362/365
threshold.

| configuration | forced top-1 agreement | threshold | qualified |
|---|---:|---:|---|
| 8192 | 353/365 | 362 | no |
| 10240 | 358/365 | 362 | no |
| 16384 | 353/365 | 362 | no |

`manifest.json` reports `qualified: false`, `long_history_status:
NOT_TESTED` and `camera_dispatch_allowed: false`. Each count is one sample of
a board characterized as non-deterministic (see the known issue above); a
re-capture may return a different count, and no count here may be quoted as
reproducible. The evidence bundle carries, per configuration, the v3 comparison
with all 365 per-step device decisions, its hash-bound independent-audit
receipt, and the sanitized provenance record.

Published assets (digests in `SHA256SUMS` and `archive-report.json`):
`camera-agent-common.tar`, `camera-agent-decode-ctx8192.tar`,
`camera-agent-decode-ctx10240.tar`, `camera-agent-decode-ctx16384.tar`,
`camera-agent-evidence.tar`, `SHA256SUMS`, `archive-report.json`.

Verify a downloaded payload with the shipped validator:

```sh
python3 validate_manifest.py --check-only   # exit 0: valid, unqualified
python3 validate_manifest.py                # exit 1: not qualified
```

## Question for the customer / 向客户的问题

**EN.** The 11 reference outputs in this handoff were generated by LiteLM-M
(checkpoint `ccdd9b95…`) under *our* 2026-09-04 tool contract (tool-set SHA-256
`4f8958ea018dd78a0df25a8365f242df8f78e0d471cff53832312c1693af059a`: `ptz_move`,
`camera_zoom`, `camera_focus`, preset and stop; no iris tool). The model
consistently emits parameters this contract does not define (`action`,
`duration`, `angle`), so the fail-closed filter rejects 10 of the 11 outputs.
That rejection is a runtime safety property of the app and will not be changed
to accept them. The one thing we need from you: **what tool contract was
LiteLM-M trained with — function names, parameter names, types, value ranges,
and the system prompt?** With it we will re-render the prompts, publish a new
golden version (never overwriting this one) and re-run the fidelity gate. Until
then this delivery claims fidelity of the OM to the HF golden only, not
compliance of the model with any camera tool schema.

**中文。**本交付件中的 11 条参考输出由 LiteLM-M（checkpoint `ccdd9b95…`）在
*我方* 2026-09-04 工具契约下生成（工具集 SHA-256
`4f8958ea018dd78a0df25a8365f242df8f78e0d471cff53832312c1693af059a`：`ptz_move`、
`camera_zoom`、`camera_focus`、预置位与停止，无光圈工具）。模型持续输出该契约
未定义的参数（`action`、`duration`、`angle`），因此 fail-closed 过滤器拒绝了 11
条中的 10 条。该拒绝是应用的运行时安全属性，不会为了接受这些输出而修改。我们
唯一需要贵方确认的是：**LiteLM-M 训练时使用的工具契约是什么——函数名、参数名、
类型、取值范围与 system prompt？**有了它，我们将重新渲染提示词、发布新版本 golden
（不覆盖本版本）并重跑保真门。在此之前，本交付只声明 OM 对 HF golden 的保真度，
不声明模型对任何摄像头工具 schema 的合规性。
