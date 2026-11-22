# Task 3 相关文献与可落地的做法（2026-11-22 10:35 整理）

**说明：** 下面的论文我只看了摘要和检索到的摘要级描述，没有通读全文；凡是来自我已有了解而非本次检索的，标了“（凭已有了解）”。每条后面写的是“对我们这道题能做什么”，都**没有经过榜单验证**。

## 现状（10:25 公开榜）

Task 3 榜首 0.48（Murmell、teambuilders），0.40–0.47 有约 7 队。我们队最高 0.2756（hu，`origin/task3_xuanhao`）。我这条线最高 0.1246，其余结论见同目录 `README.md` 与 `RESULTS.md`。

我们这道题的特殊点：**没有保留集，只有原模型和 3500 张遗忘图；评分逐图比较与“重训模型平均分布”的 KL**。文献里绝大多数方法假设有保留集。

## 一、直接对应“让遗忘图表现得像没见过的图”

**Reference-Guided Machine Unlearning（ReGUn，Mirlach 等，arXiv 2603.11210，ICLR 2026 workshop）**
- 做法：不把遗忘图“推错”，而是把遗忘图上的预测蒸馏到**同类的留出图**上原模型的预测；参照按类（或按样本）配对，并对齐类别直方图。
- 与我们的关系：思想和 hu 的目标、以及我做的 5 折 `t_world` 是同一个——“遗忘图应当像同类里没见过的图”。它用真实留出图当参照，我们没有，所以只能用交叉拟合去造。
- 可落地：hu 的目标里已有“类级先验 R_c”。可以把 R_c 换成 `t_world` 在该类上的平均分布（`gold_standin_world.npz` 里按 `label` 分组求 `p_T2_mean` 的均值），作为每张图分布的对冲项。几秒钟的事，不需要重训参照。

**Decoupled Distillation to Erase（arXiv 2503.23751）**
- 面向“类”的遗忘：把要忘的类的 logit 与其余 logit 解耦后分别蒸馏。
- 可落地：对应 hu 对 9 个封顶类的处理（标签单独压低 + 其余类分布单独拟合）。他的 `--gap_full` hinge 加 KL 已经是这个结构。

## 二、只用遗忘集（无保留集）的方法

**Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization / An Information Theoretic Approach to Machine Unlearning（JiT，Foster 等，arXiv 2402.01401）**
- 做法：只用遗忘样本，最小化模型输出对该样本邻域扰动的敏感度（让遗忘图和它的扰动副本输出一致、平滑）。
- 与我们的关系：评分有一半在“隐藏的改动副本”上。我今晚的三次同目标提交给出同一结论：裁剪副本上忘得越多分越高（0.092 → 0.125 → 0.148）。
- 可落地：在遗忘侧加一项**原图与改动副本之间的一致性损失**（两者输出的对称 KL），而不只是各自去拟合目标。hu 现在是“每个视图各自拟合同一目标”，加一致性项成本很低。

**SCAR（Bonato 等，arXiv 2404.12922）**
- 做法：无保留集。用马氏距离把遗忘样本的特征推向最近的其它类的特征分布；用**分布外图像**上对原模型的蒸馏来保住其余行为。
- 可落地：(1) “最近的其它类”给出了 9 个封顶类的图该把概率分给谁的另一种估计——用第 7 层特征到各类中心的马氏距离，可与 `t_world` 的分布互相印证；(2) 分布外图像当保留锚点。但我们的公开分显示保留项不是瓶颈，这一条优先级低。

**Towards Source-Free Machine Unlearning（arXiv 2508.15127）、GFOES（AAAI）**
- 检索到但未细看，都是不依赖原训练数据的设定。

## 三、记忆在网络里的位置（支撑“只改后几层”）

**Can Neural Network Memorization Be Localized?（Maini 等，2023）**（凭已有了解）
- 记忆并不集中在某一层，而是分散在少量神经元上；可按样本定位。

**Improved Localized Machine Unlearning Through the Lens of Memorization（DEL，Torkzadehmahani 等，arXiv 2412.02432）**
- 做法：按“对记忆最关键”的标准选一小部分参数，只重置/微调这部分。
- 可落地：现在我们训练的是 block 8–11 的全部参数。可以只动其中对遗忘图梯度最大的那部分通道，理论上对其它图的连带影响更小。但同上，保留项不是瓶颈，收益可能有限。

**Localizing Memorization in SSL Vision Encoders（Wang 等，arXiv 2409.19069，NeurIPS 2024）**
- 结论：ViT 的记忆主要在更深的 block，且主要在全连接（MLP）子层。
- 与我们的关系：和队友“第 8 层读出还没记住、第 10 层已经记住”的测量一致（我这边：从第 8 层读出与记住标签一致率 0.50，第 9 层 0.79，第 10 层 0.99）。
- 可落地：只训练 block 8–11 的 MLP 子层、冻结注意力子层，是一个便宜的对照。

**Weight-space memorization decomposition（arXiv 2510.24256）**
- 用近似 Hessian 的特征基把权重分成“记忆方向”和“泛化方向”，直接在权重上抑制记忆。无训练。未细看，时间上来不及实现。

**Benchmarking Unlearning for Vision Transformers（Zhao、Luca、Triantafillou，arXiv 2602.20114）**
- ViT 上遗忘方法的基准，重点是利用记忆信息的方法。可作为赛后复盘的参照。

## 四、有保留集的经典方法（官方参考文献，在本题上受限）

- **SCRUB**（Kurmanji 等，NeurIPS 2023）、**SalUn**（Fan 等，ICLR 2024）、**NPO**（Zhang 等，2024）、**Zero-Shot MU**（Chundawat 等，2023）。
- 前三者都要在保留集上拉回；队友 li103 试过无保留集的 SCRUB 变体，没有超过当时的最高分。Zero-Shot MU 的“无知老师”目标接近均匀分布，相当于把准确率压到 0——我提交过一个准确率为 0 的目标，得 0.043，远差于保留约一半准确率的目标。

**Are we making progress in unlearning? Findings from the first NeurIPS unlearning competition（Triantafillou 等，arXiv 2406.09073）**
- 那次比赛同样以“与重训模型的输出分布有多接近”计分。论文结论之一：榜首方案在该评测下超过了已发表的方法，且超参需要针对数据集重调。
- （凭已有了解，未在本次检索中核实）当时靠前的方案大多是“重置或扰动一部分层 → 在保留集上微调”，有的在遗忘集上把输出拉向某个平均分布。前半部分和我做的“重置第 7–11 层再训练”同源，后半部分需要保留集。

## 五、如果只剩一两次尝试，我的排序

1. **原图与改动副本的一致性项**（JiT 的思路）——有我们自己的公开分支持“改动副本上的遗忘很重要”，实现最简单。
2. **9 个封顶类的图，概率分给谁**：把 `t_world` 的分布（或它的类平均）掺进 hu 现有目标的封顶类部分（ReGUn 的思路）。不要整体替换其余 36 类——我的目标在那部分偏轻（损失 1.85 对他约 2.8）。
3. 其余（只训 MLP 子层、马氏距离、权重分解）留作赛后复盘，时间上来不及验证。

## 来源

- [Reference-Guided Machine Unlearning](https://arxiv.org/abs/2603.11210)
- [Decoupled Distillation to Erase](https://arxiv.org/html/2503.23751v1)
- [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://arxiv.org/abs/2402.01401v2) / [An Information Theoretic Approach to Machine Unlearning](https://arxiv.org/html/2402.01401v4)
- [SCAR](https://arxiv.org/abs/2404.12922v1)
- [Towards Source-Free Machine Unlearning](https://arxiv.org/pdf/2508.15127)
- [Improved Localized Machine Unlearning Through the Lens of Memorization](https://arxiv.org/html/2412.02432v1)
- [Localizing Memorization in SSL Vision Encoders](https://arxiv.org/abs/2409.19069)
- [Weight-space memorization decomposition](https://arxiv.org/pdf/2510.24256v1)
- [Benchmarking Unlearning for Vision Transformers](https://arxiv.org/html/2602.20114v1)
- [Are we making progress in unlearning?](https://arxiv.org/abs/2406.09073)
- [Forget Vectors at Play](https://arxiv.org/pdf/2412.16780)
