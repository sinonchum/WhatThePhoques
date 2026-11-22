# 交接：Euro AI Championship Paris · Task 2 Model Equality（队伍 WhatThePhoques）

> **2026-11-22 12:10（巴黎时间）更新：比赛提交已关闭。** 公开分最终是 1.00，留出分尚未公布。最终提交在 `task2/final/`，方法和证据在 `task2/experiments/fingerprint/README.md`，逐步的判断在同目录的 `DECISIONS.md`。下面是 11-21 22:00 的交接原文，其中的分数、额度、队列状态和「建议的下一步」都已过时；「不要做的事」仍然有效。

写于 2026-11-21 22:00（巴黎时间），由上一个 Claude 会话交给 Claude Code。目标：把 Task 2 公开分从 0.10 提到超过榜首的 0.75。**提交截止 2026-11-22 12:00（巴黎时间）。**

用户的原话要点：之前 Codex 的版本「很不好」，要创造新方法；查询正式端点和上传榜单的权限都已给出。

## 不要做的事

- 不要把队伍密钥写进任何文件、提交或输出。它在用户终端的环境变量 `TASK2_API_KEY` 里。
- 不要 `git worktree prune`。主仓库还挂着另一个 worktree（`hermes/task2-offline`）。
- 不要改主仓库 `../euro-ai-championship-paris` 的工作区和 `task2_xuanhao` 分支。
- 上传每 30 分钟只能一次，失败也算冷却。榜单只保留最高分。

## 位置

| 内容 | 位置 |
|---|---|
| 本 worktree | `outputs/eac-task2-claude`，分支 `claude/task2-g2-signal`，基于 `71be00d` |
| 主仓库（勿动） | `outputs/euro-ai-championship-paris`，分支 `task2_xuanhao` |
| 题面整理 | `docs/tasks/task2-zh.md`，原 PDF 在同目录 |
| 第一轮缓存实验（已提交 `63a61b4`） | `task2/experiments/persistent-agreement/README.md` |
| 新方法的工具与数据（进行中） | `task2/experiments/fingerprint/` |
| 历史缓存 | `task2/test-data/README.md`；`python3 task2/test-data/verify.py` |

远端没有核对过：上一个会话的沙箱连不上 GitHub。本分支只在本地，没有推送。

## 比赛接口

- 服务器 `http://34.147.229.55`，任务号 `33-model-equality`，请求头 `X-API-Key`。
- `GET /33-model-equality/v1/models` 返回每端点额度；`POST /33-model-equality/v1/chat/completions`（`model`、`max_tokens` ≤ 256、单条 user 消息）；`POST /submit/33-model-equality` 上传 CSV。写法见 `task2/.starter/`（主仓库里有，已 gitignore）。
- **每条请求固定约 30–36 秒，与回答长度无关。** 全队最多 4 个并发，超出返回 429 "Your team already has 4 requests open"。所以一分钟最多约 7 条，每条回答要尽量多装信息。
- 回复体只有文本、`finish_reason`、`budget`，没有 token 用量。
- 额度（21:30 实测）：正式端点已用 44–45 / 100，各剩 55–56 条。练习端点 `ref-01..09` 无上限。
- 交接时曾有另一个 harness 同时占用并发并消耗正式额度，用户被提醒过要停掉，是否已停未确认。动手前先查余额。

## 榜单与历次上传

21:30 的 Task 2 公开榜：Fable 4.2 0.75；3V1B、Azerit、Eugenia、Holmes 0.70；CISPA fan boys 0.60；Murmell 0.55；SLZ 0.20；我们 0.10。榜上分数都是 0.05 的倍数。

我们的四次上传：18:04 得 0.05（4 个家族），之后 18:40、19:35、20:24 都是 0.10（3 个家族）。0.10 就是家族项满分、模型项约为零。

当前采用的家族划分（按首轮答案的经验规则得到，公开分支持它是对的）：

- G1：ep-01、02、04、12、14、17、18、23、24、25、33。较强的模型，回答几乎都只有一个字母或数字。
- G2：ep-03、06、07、08、20、21、22、26、28、30、32。中等。
- G3：ep-05、09、10、11、13、15、16、19、27、29、31。很弱，风格接近练习端点。

练习端点真值：ref-01 原版、03 系统提示、05 persona、06 水印、08 温度 1.0 采样是同一个模型；ref-02 FP8 量化、09 LoRA 0.3、04 LoRA 0.7、07 LoRA 1.5 各自不同。练习模型自称 "trained by Hugging Face"，是 SmolLM 一类的小模型。

## 已经确认的事实

1. **Codex 的思路在正式数据上不成立。** 它把格式异常的端点拆出去，其余合成一大组。按题面评分，如果大组基本是一个模型，公开分应在 0.4 左右；实际 0.10。推断：每个家族里有多个模型，很可能是多个「一模型多部署」的小组，而不是一个大组加若干单点。这是推断，没有标签。
2. **单字母答案的相似度是巧合水平。** 660 条缓存上的打乱检验见 `persistent-agreement/README.md`。
3. **重复 5 次后的数据**（`fingerprint/data/hermes-repeats/formal_probe.jsonl`，33 端点 × q02、q13、q16、q23、q26、q30 × 4 次，加缓存里的 1 次）：
   - G2 里同一端点自身的重复一致率与不同端点之间几乎相同（q16 0.76 对 0.69，q26 0.39 对 0.41，q02 0.19 对 0.21）。端点是带温度采样的，单次答案是噪声。
   - G1 的 q13 有真实差异：ep-23 五次全 B；ep-24 是 ABBAB；ep-02 是 BBAAA；ep-04、12、17、25、33 五次全 A。
   - G1 的 q02：ep-04、12 五次全是 100，ep-18 四次全是 100；ep-33、17、14、25、02 各有两次别的数字（40、32、160、146 等），像是更高的温度或水印。
   - G3 的 ep-10 自成一派：q02 三次 1003，q30 五次全 A（别人多为 B），q16 四次 C。
4. **练习端点上的新探测题**（`fingerprint/queue/results/010-ref-battery-a/answers.jsonl`，交接时第一轮 17 种题跑到 mc1）：
   - `pa1`（让模型复述事先收到的指令）让 persona 端点 ref-05 说出了角色设定；系统提示端点 ref-03 没有复述。小模型大量编造，G1、G2 这种较强的模型预计更听话，未验证。
   - LoRA 0.7 和 1.5 的回答爱加句末句号（"100."、"…40."），并重复 "trained by Hugging Face, trained by Hugging Face"。这是练习 LoRA 的特征，正式家族的 LoRA 未必相同。
   - 自由写作题（fg1–fg4）各端点差别很小，不能直接当指纹。
   - 小模型不会给随机数（rc1、rc2 基本是 1, 2, 3, …）。
5. 回答里用第三人称提到 the user / the human：练习数据只有系统提示端点出现过；正式数据里 ep-20 一次、ep-16 两次。ep-20、ep-16 很可能带系统提示，按题面这不改变模型身份。

## 新方法的设计（尚未在任何端点上验证）

思路：逐对比较端点的「决策指纹」，而不是找离群点。

- `fingerprint/batteries.py`：两套各 40 道二选一题（`s1`、`s2`），一条回答给出 40 个决策。每套有两个朝向（`s1a` 原样，`s1b` 选项互换），`parse()` 把字母换算成「是否选了第一项内容」。
- 为什么这样设计：
  - 温度只缩放 logit，不改变偏好的方向，所以比较偏好向量的相关性或众数，而不是绝对频率。
  - 水印和系统提示对某个字母的固定偏好，在两个朝向相减时抵消。
  - LoRA 改的是内容偏好，应当留下；FP8 量化几乎不动 logit，大概率分不出来（榜上 0.70–0.75 很可能就是这个上限）。
- 计划的用量：每端点 4 个提示 × 5 次 = 20 条，每题 10 个样本；再加 `pa1` 2 条。G3 的小模型可能无法按格式答 40 行，需要先试，不行就换短题组或回到练习端点上验证过的格式特征。
- 聚类：家族内按偏好向量的相似度做层次聚类，阈值用同一端点两半数据之间的相似度来标定。

## 转发脚本（可选）

上一个会话的沙箱连不上比赛服务器，所以写了 `fingerprint/bridge.py`：用户在终端里运行，它执行 `fingerprint/queue/jobs/*.json` 里的任务，把结果写到 `queue/results/<任务名>/`。**Claude Code 在这台 Mac 上有网络，可以直接调用接口，不必再用它。** 如果继续用：

```bash
python3 task2/experiments/fingerprint/bridge.py            # 需要环境变量 TASK2_API_KEY
python3 task2/experiments/fingerprint/queue_job.py --help  # 写任务文件
```

交接时的队列状态：
- 终端里跑着的是旧版进程（pid 91860），正在执行 `010-ref-battery-a`（9 个练习端点 × 17 题 × 3 次）。旧版不支持插队和取消。按 Ctrl-C 重启后才是新版。
- 已排队未执行：`005-scored-pilot`（ep-12、ep-21、ep-05 各问 `s1a` 和 `pa1`，共 6 条正式查询）和 `006-ref-items`（练习端点 × `s1a`、`s1b`，18 条）。
- `011-ref-battery-b`（765 条练习查询）已放了取消标记，新版会跳过。
- `queue/` 已 gitignore。有用的结果请拷到 `fingerprint/data/` 再提交。

## 建议的下一步

1. 查余额；确认另一个 harness 已停。
2. 跑 6 条正式试探，看三个家族能否按格式回答 40 行，以及 `pa1` 在较强模型上是否直接说出系统提示。
3. 在练习端点上跑 `s1a`、`s1b` 各若干次，检查同模型的五个端点（01、03、05、06、08）是否比 LoRA 端点更接近。练习模型很弱，这一步可能没有信号，不要因此否定对 G1、G2 的方案。
4. G1、G2 全量采集：22 个端点 × 4 个提示 × 5 次 = 440 条，约 1 小时。先各跑 1 次看解析率再放量。
5. G3 单独设计。
6. 聚类、与历次上传去重（`persistent-agreement/dedupe.py`）、上传。每个家族可以单独验证：只对一个家族给出分组、另外两个家族全部判为不同，公开分的变化就只反映这个家族。题面反复警告不要只对公开分调参，公开分只当粗检查。

## 复现第一轮实验

```bash
cd task2/experiments/persistent-agreement
python3 agreement_test.py --out content_persistence.json
python3 format_persistence.py --out format_persistence.json
python3 replay_compare.py --out replay.json
python3 make_candidate.py && python3 dedupe.py candidate-replicated.csv
```

需要 numpy 和 scipy（`task2/requirements.txt`）。`fingerprint/` 下的脚本只用标准库。
