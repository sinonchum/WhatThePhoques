# Task 2 历史测试数据

新 harness 可以从下面两份 JSONL 开始。它们合计包含 **1470 条不同的真实回答**，每行都有原始题干、端点编号、题号、重复序号和完整回复。运行离线实验只需克隆本分支。

| 用途 | 文件 | 覆盖 |
|---|---|---|
| 有监督开发、重放 | [practice_labeled.jsonl](collected/next-round/practice_labeled.jsonl) | 9 个练习端点 × 30 题 × 3 次，共 810 条，已关联官方 family、model 和 variant_kind |
| 正式推断、无监督对照 | [combined_formal_unlabeled.jsonl](collected/official-round2/combined_formal_unlabeled.jsonl) | 33 个正式端点 × 20 题 × 1 次，共 660 条 |
| 对照题型和标签 | [30 道原题](../prompts.csv) · [9 个练习端点真值](../checkpoint/practice_answers.csv) | 题库的 dev 和 holdout 标记划分的是问题。不同重放会复用相同的练习端点 |

读取整批正式数据时，只需使用合并文件。如果要按时间分开评测，读取[首轮 330 条](collected/next-round/formal_unlabeled.jsonl)和[第二轮新增 330 条](collected/official-round2/new_formal_unlabeled.jsonl)。这两份按顺序拼接，与合并文件逐字节一致。带元数据的 JSONL 和 [checkpoint 原始日志](../checkpoint/) 使用相同的回答；跨导出文件去重时使用 `(dataset_split, endpoint, id, repeat)`。

## 历史实验与评分

| 想查什么 | 在哪里 |
|---|---|
| 解析器修改前后的练习预测、错对明细 | [原版](collected/practice-original/) · [改进版](collected/practice/) |
| 实际提交的 CSV、当次公开分、榜单快照 | [首次提交](collected/first-live/) · [330 条的第二版](collected/offline-v3/) · [660 条的提交](collected/official-round2/) |
| 家族划分、端点对距离、聚类候选和模型对照 | [历史分析](analysis/) · [离线 v2](collected/offline-v2/) |
| 按种子重放、困难题、变体消融与失败实验 | [稳定性复核](collected/variant-stability/) · [样式和短答案实验](analysis/transfer3/) · [最近一次候选及其对照](../experiments/choice-edit/) |
| 验证读写、评分和提交流程的合成样本 | [demo](collected/demo/) 有 540 条合成练习回答和 1980 条合成正式格式回答 |
| 设计过的备用题单 | [10 个备用问题包](proposed/prompt-bundles.json)，每包有 3 个选择项任务。现有真实回复对应上表的 30 道原题 |

正式端点文件里的候选分组都是算法预测。公开分只对应一份完整提交，逐端点的官方标签仅随练习数据提供。

[机器可读索引](manifest.json) 列出每份归档数据的仓库相对路径、原始位置、文件字节数和 SHA-256。历史实验的原始副本和对应导出均已保留，内容完全相同的文件可按哈希去重。仓库根目录运行以下命令可验证全部索引文件，同时检查真值关联、每个端点的覆盖、两轮拼接以及回复与原始日志的一致性。

```bash
python3 task2/test-data/verify.py
```
