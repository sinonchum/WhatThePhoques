# Task 2 · 从回答到端点分组

目标是用同一批题比较 33 个端点，输出 `endpoint,family,model`。家族分占 10%，同一真实家族内的模型身份分占 90%。水印、系统提示和解码变化保持模型身份；量化配置或 LoRA 变化会改变身份。[官方题面](../docs/tasks/Model_Equality_Testing.pdf)

## 接手当前版本

第二轮提交使用 33 个正式端点各 20 条回答，共 660 条。2026-11-21 巴黎时间 19:35，官方回执记录的公开分为 **0.10**，当时排名 **6 / 20**。这份提交得到了 3 个候选家族、13 个候选模型身份。正式端点的身份尚无公开真值。[提交 CSV](checkpoint/submission.csv) · [官方回执快照](checkpoint/receipt.json)

在仓库根目录运行以下命令，可以用缓存重新生成当时实际上传的 CSV。离线运行使用 Python 3.9–3.12，依赖版本保存在 [requirements.txt](requirements.txt)。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r task2/requirements.txt
python task2/variant_rank.py --answers task2/checkpoint/formal.jsonl --families task2/checkpoint/families.csv --out task2/runs/reproduced.csv
cmp task2/runs/reproduced.csv task2/checkpoint/submission.csv
```

[variant_rank.py](variant_rank.py) 在给定家族内逐题比较回答格式、选项复述和解释顺序，结合偏差大小和该题内的相对排名，再合并核心组、拆出偏差较高的端点。当前采用上三分之一的经验切分。[家族表](checkpoint/families.csv) 来自首轮回答的经验规则：q03 回复超过一个词归入 G3，其余按 q07 整数的绝对值小于 10 或至少 10 分入 G1 或 G2。可以单独替换家族表，比较两个阶段各自带来的变化。

### 新算法如何对照

[练习缓存](checkpoint/practice.jsonl) 有 9 个端点 × 30 题 × 3 次重复，配套[公开标签](checkpoint/practice_answers.csv)。[正式缓存](checkpoint/formal.jsonl) 有 660 条无标签回答。公开分是对一份提交的整体反馈。训练或选阈值时请区分练习真值与正式预测。

把新算法放在 `task2/my_solver.py`，实现 `predict(rows)`，返回字典 `{匿名端点: 模型组标签}`。每次重放会打乱端点名与行序，从五类题各抽两道，每个端点每题抽一条回答。使用相同种子可以逐次对照，过线定义为练习 **model MCC 严格大于 0.65**，目标为至少 1434 / 2048 次过线。

```bash
python task2/replay.py --solver variant_rank --seed 731029
python task2/replay.py --solver variant_rank --seed 519883
python task2/replay.py --solver variant_rank --seed 519883 --protocol fixed-hard
# 将 --solver 换成 my_solver 可评估自己的实现
```

| 抽样协议（每组 2048 次） | 种子 731029 | 种子 519883 |
|---|---:|---:|
| 五类题分层抽十题，过线次数 | 1547（75.5%） | 1538（75.1%） |
| 固定另一组十道开发题，过线次数 | 831（40.6%） | 880（43.0%） |

两组种子都使用同样的 9 个练习端点，且都属于同一真实家族。固定正式两轮采用的 20 道题在练习缓存上重放时，FP8 与原版仍有 1018 / 1024 次被误合并。这是接手后值得优先验证的端点对。[逐种子结果一](checkpoint/verification-731029.json) · [逐种子结果二](checkpoint/verification-519883.json) · [20 题重放与变体错误](checkpoint/practice-replay.json)

## 早期众数与分布基线

| 步骤 | 做什么 | 可以检查什么 |
|---|---|---|
| 选题与采集 | [题库](prompts.csv) 有 30 道题，覆盖数学、代码、语言、专业知识、逻辑五类。每类 4 道开发题、2 道不同的留出题。官方客户端把每次回复追加到 JSONL。 | 每个端点及题号采了几次，追加一轮还要多少额度 |
| 解析答案 | [解析器](inputs.py) 把具体选项或整数归一化。两个端点选错时，仍保留它们各自选了哪个答案。无法判断的回复记为缺失。 | 每个端点的未解析数、每对端点有多少共同有效题 |
| 比较接口 | [算法](clustering.py) 的 `mode` 只比较最常出现的答案，并列众数共享权重。`js` 比较同一道题上的完整答案频率，用以 2 为底的 Jensen–Shannon 距离。每一类内对题取平均，再让各类等权。 | 同样的采样预算下，保留频率是否比只看众数更容易区分端点 |
| 选模型阈值 | [compare](pipeline.py) 在 20 道开发题上分别选择模型分组阈值，再在 10 道留出题上报告 MCC、误合并和误拆分的具体端点对。自动选方法只看开发分，分数相同时选 `mode`。 | `comparison.json` 和两份留出题预测 CSV |
| 生成分组 | 对正式端点做平均链接层次聚类，先按较宽的距离阈值划分家族，再在家族内按更严的阈值划分模型。 | `submission.csv` 有 33 行数据，`distances.json` 给出全部 528 对的距离、题目覆盖和每类贡献 |

官方 9 个练习端点都属于同一家族。练习时将家族视为已知，只比较模型身份。开发题上的模型阈值从 0 到 1 按 0.1 的步长选择。**家族阈值需要在正式数据上做敏感性分析。** 可以用相同回复试多个合理阈值，查看哪些端点在阈值略变时就换组，优先复查这些端点。下文的家族阈值是示例，应取不低于开发题所选模型阈值的值。[练习真值](https://huggingface.co/datasets/SprintML/model_equality/blob/main/practice_answers.csv)

## 先跑完全离线的例子

在仓库根目录运行以下命令，使用 Python 3.9 或更高版本即可。演示的回答和标签都是合成数据，分数只用于验证程序。每次新建演示时用一个新目录。

```bash
python3 task2/pipeline.py demo --out task2/runs/example
python3 task2/pipeline.py compare --answers task2/runs/example/practice.jsonl --truth task2/runs/example/truth.csv --out task2/runs/example/check
python3 task2/pipeline.py predict --answers task2/runs/example/scored.jsonl --calibration task2/runs/example/check/comparison.json --family-threshold 0.65 --out task2/runs/example/pred
```

先打开 `check/comparison.json` 看两种方法，再打开 `pred/submission.csv` 看最终格式。模拟数据专门造了容易识别的模型差异，实际比赛分数要用真实回复衡量。

## 换成比赛数据

本次本地副本已经把主办方的工具下载到 `task2/.starter`。在其他机器上可复用队长准备的 `model_equality` 目录，将下列命令里的 `--kit` 和真值路径改到那里，或克隆[官方起始代码](https://huggingface.co/datasets/SprintML/model_equality/tree/main)。联网采集还需安装官方客户端使用的 `requests`，并在当前环境设置 `TASK2_API_KEY`。

**练习。** 运行第一条会打印本地计划；在同一条命令末尾追加 `--execute` 才会发请求。第二条用实际采集的练习回答选阈值。第三条可用官方评分器独立核对 `mode` 的留出题得分，也可替换为 `holdout-js.csv`。

```bash
python3 task2/pipeline.py collect --group practice --kit task2/.starter --answers task2/runs/practice.jsonl --repeats 3
python3 task2/pipeline.py compare --answers task2/runs/practice.jsonl --truth task2/.starter/practice_answers.csv --out task2/runs/practice
python3 task2/.starter/score_practice.py task2/runs/practice/holdout-mode.csv task2/.starter/practice_answers.csv
```

**正式查询预算预览。** 第二轮回执时每个正式端点累计使用 20 / 100 条。下面直接读取快照，预览复用已采十题时需要的新增条数（应为 0）。之后如要新增采集，先确认队伍最新的共享日志，使用包含这 660 条及之后所有回答的可写副本。保留快照供离线复现。

```bash
python3 task2/pipeline.py collect --group scored --kit task2/.starter --answers task2/checkpoint/formal.jsonl --ids q02 q03 q07 q08 q13 q16 q20 q22 q25 q26
```

全队共用同一份正式回答日志。可用 `--endpoints ep-01 ep-02` 将后续复核限制在指定端点。并发和服务器剩余额度由官方采集客户端处理，本地预览只统计指定日志。

**正式导出。** 用真实练习报告配合真实正式回答导出 CSV。加 `--method js` 可对比另一版分组，输出到另一个目录。提交由队员核对 CSV 后使用官方 `submission.py` 完成。

```bash
python3 task2/pipeline.py predict --answers task2/runs/scored.jsonl --calibration task2/runs/practice/comparison.json --family-threshold 0.95 --out task2/runs/formal
```

## 研究依据

这版比较的是两条可在同一份比赛回答上复验的路线。LLMmap 的已知模型识别实验支持优先筛选互补的问题；短答案指纹研究给出了用分布距离比较端点的思路。AgentProv 的量化实验中，三种方法都漏掉了两组 8 位量化对照，练习时应该单独看量化端点是否被误合并。[LLMmap](https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini) · [短答案分布指纹](https://arxiv.org/abs/2607.10252) · [AgentProv，表 14](https://arxiv.org/html/2609.00052v1#A1.SS8)
