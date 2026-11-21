# 两次重复回答的公开方法复刻

复用了 [EuropeanAiChampionship 公开分析脚本](https://github.com/gokunguru/EuropeanAiChampionship/blob/3440900b6fee2f2c436affea4020e6a81cee1936/src/model_equality/ms_battery_analyze.py) 的答案提取、去除端点自身波动的距离、平均链接聚类和四个阈值。仓库与公开榜首队伍的对应关系未得到验证。原项目30道题里有26道进入聚类，4道随机题作对照。我们缓存的30道题与它的题干逐字重合数为0。此处检验公开算法在本队题库的可迁移性。没有新增模型调用或榜单提交。

## 同样两次回答、同样十道留出题

9个练习端点各答30题，每题3次。从同一份缓存分别取重复 (0,1)、(0,2)、(1,2)。三种组合共享端点、问题和部分回答。先在20道开发题上，以三种组合的平均截断 model MCC 从 0.05、0.10、0.15、0.20 四个阈值里选取一个，再冻结阈值评估10道留出题。在两种解析方式下，开发分四档全部为0；按原脚本阈值顺序取第一档0.05。分数均为九个参考端点全部36对上的模型 MCC。

| 方法 | (0,1) | (0,2) | (1,2) | 平均 |
|---|---:|---:|---:|---:|
| 原版末整数／末词解析 + 原版聚类 | 0.194 | 0.175 | 0.000 | 0.123 |
| 选择题优先提取选项，提取失败则回退原解析 | 0.127 | 0.037 | 0.000 | 0.055 |
| 本队已有 choice_edit，同样的两次采样 | 0.734 | 0.734 | 0.734 | 0.734 |

原版把系统提示版 ref-03 和原版 ref-01 错拆3/3次；正确拆开弱 LoRA ref-09 和 FP8 ref-02 各2/3次。本队格式基线正确合并系统提示版3/3次，正确拆开弱 LoRA 3/3次，漏掉 FP8 3/3次。九个留出题预测均另经官方 score_practice.py 核对。

阈值敏感性较大：事后挑最有利的留出题阈值，原解析平均最高为0.413（阈值0.10）。该值使用了留出标签，不属于开发阶段选出的结果。此前的本队方法也使用过这些练习端点，不可用这场对照推算正式公开分。

[practice.json](practice.json) 保存全部阈值的分组、混淆计数与目标端点判断。上游主程序用距离阈值聚类，虽定义了分题型置换检验函数，但主程序没有用 p 值决定合并。我们另对三个目标端点与原版逐题型运行原函数（2000次置换），并计算同一统计量的精确置换尾概率。见 [pair-tests.csv](pair-tests.csv)。每题重新分配四条回复的检验假设要求回答可交换，不直接等价于比赛的模型身份定义。

## 复现

从队伍仓库根目录运行。克隆用于取得上游公开源码，重放脚本自身只读取本地文件，并在导入上游模块前核验三个源文件的 SHA-256。

```bash
git clone https://github.com/gokunguru/EuropeanAiChampionship /tmp/eac-battery-source
git -C /tmp/eac-battery-source checkout 3440900b6fee2f2c436affea4020e6a81cee1936
python3 task2/battery_replay.py --source /tmp/eac-battery-source/src/model_equality --out task2/runs/battery-reproduction
cmp task2/runs/battery-reproduction/practice.json task2/experiments/battery-replica/practice.json
```
