# Task 3 遗忘研究：20 篇论文按提分优先级排序

**只跑一轮，我选第 1、2 篇的组合。** 从已有公开分 **0.2756** 的 `ex1413` 出发，对两套教师取得共识的图片，调整它最像哪个其他类别。训练目标里原伪标签的概率、每张图的交叉熵和预测熵都可以精确保住。**若能跑第二轮，选第 3 篇**，让学生和教师看完全相同的轻裁剪。[最好提交回执](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/submissions.jsonl#L39) · [现有训练配方](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/README.md)

排名依据是从当前最好模型再提高**官方总分**的机会。能直接影响逐图预测、已有队内证据、能用现有文件做对照的排前面。20 篇中有些回答同一个实验的不同环节。以下分数来自仓库截至 2026-11-22 的回执，新增分析来自已保存的预测，两项推荐都还没有官方得分。

## 为什么先改错类

官方给 3,500 张无标签遗忘图，比较精确原图和隐藏改动副本，两组等权。忘没忘够，看真实标签交叉熵的均值和标准差；概率分给了谁，看温度 2 下逐图的 `KL(gold || sub)`。还要保住其余图的准确率和预测分布，独立验证集准确率低于 70% 时总分为零。下文的“原伪标签”始终指原模型在干净图上的首选。[官方题面，第 1–3 页](Unlearning.pdf)

按原伪标签计数，9 类各出现 129 或 130 次，共 1,169 张，简称“封顶类”。另有 36 类共 2,331 张，其余 55 类在遗忘集里没出现。团队对封顶图加大原伪标签和最佳替代类别之间的分差时，一次提交从 **0.1861 到 0.2509**，该轮也改了训练和分类头缩放，无法把全部增益归给分差。已有的 `xh_target6.py` 则尝试按类别数量重新分配其他类别的概率，并固定温度 2 的原类概率。下面的实验多控制一个变量：**温度 1 的逐图伪标签交叉熵也精确固定**。[提交记录](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/README.md#L48-L96) · [现有目标代码](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/xh_target6.py#L25-L39)

### 用已有预测做的对照

`f7` 重置原模型的第 7–11 个 block，用五折训练后几层，每张图只取未见该图的那一折，最后平均 5 个种子。`t_world` 额外从训练输入和教师输出中排除全部 9 个封顶类，平均 2 个种子。两者共用见过这些图的浅层。把二者各自的温度 2 概率限制在其余 91 类并重新归一化后，针对 1,169 张封顶图做了以下比较。[两份预测及生成方式](https://github.com/sinonchum/euro-ai-championship-paris/tree/00a7a8d26f6d093bdc66a1ae90c970d6957cf90b/task3_claude)

| 用什么预测另一套教师 | 以 `t_world` 为参照的平均 KL ↓ | 以 `f7` 为参照的平均 KL ↓ |
|---|---:|---:|
| 同伪标签的其他图片取均值，排除本图 | 1.2021 | 1.1776 |
| 本图和上述均值各占一半 | 0.4560 | 0.4281 |
| 只取本图的预测 | **0.2785** | **0.2959** |

逐图方法在 **9 个封顶类分别比较时都胜出，两个方向都是如此**。两套教师在 91 类里首选相同的有 **823/1,169 张（70.4%）**。放开其余 8 个封顶类、只排除原伪标签时，一致数降为 **374 张（32.0%）**。因此已有证据最支持的是**在那 91 类内部决定哪类更像**。屏蔽全部封顶类是 `t_world` 的构造假设，教师间的 KL 也只衡量共识，不能当官方 KL 或预测涨分。

### 前两轮怎么做

| 优先级 | 相对同条件对照只改什么 | 用什么决定是否继续 |
|---|---|---|
| 第一轮，逐图选替代类 | 两版从同一 `ex1413` 权重出发，对照用它在干净图的输出作目标。实验版只在上述 823 张里操作：若教师共同首选与当前目标在 91 类内的首选不同，交换这两个类的 logits。其余图片、保留项、步数和随机增强完全相同。 | 在团队已有小型重训实验中按同样规则构造两版，比较官方公式的代理总分。单看预测项可诊断替代类是否选得更好，纳入损失项和保留项后才能决定选哪版。 |
| 第二轮，同裁剪教学 | 单独从相同起点做对照。轻裁剪输入记作 `v`，用冻结的原模型读出 `block 8`，取 `z8(v) − z8(原图)` 加在原训练目标上。教师和学生使用同一批裁剪，保留侧照旧。 | 在训练中没用过的一批轻裁剪上算与小型重训模型的逐图 KL。如果第 1、3 名对应的两轮各有收益，再考虑第 4 名的权重平均。 |

第一轮的关键是“交换”。它保留所有 logit 的数值集合，且原伪标签不在可交换的 91 类里，所以任何温度下原伪标签的概率、整张图的预测熵、9 个封顶类的总概率都保持相同。这个性质对构造的目标成立，训练完成的学生仍须评估。那里没有外部数据或新模型依赖。小型重训实验只用来比较方案，其诊断用的外部初始化不带入正式提交。[小型重训实现](https://github.com/sinonchum/euro-ai-championship-paris/blob/00a7a8d26f6d093bdc66a1ae90c970d6957cf90b/task3_claude/modal_app.py#L539-L568)

## 论文排名

### 1. Attribute-to-Delete: Machine Unlearning via Datamodel Matching

**Georgiev 等，ICLR 2025。** 作者先估计删除一组样本后模型在每个输入上的输出，再让原模型拟合它。图 8 的非随机删除实验里，只估计样本对自身的影响，丢失了样本之间的作用，整体放大或缩小也补不回来。这正好解释为什么只调全局温度不解决“判成谁”。原文依赖全训练集做数据归因，我们可以用现成的 `f7` 和 `t_world` 给出两个逐图估计。列第一的依据是上面的双向 KL 对照。[论文，第 5.2 节、图 8](https://arxiv.org/html/2410.23232v2#S5.SS2)

### 2. Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks

**Zhou 等，CVPR 2025。** DELETE 用掩码把要忘的类与其余类分开优化，无须保留图。表 1 的 CIFAR-100 单类删除实验中，删除类测试准确率降到 0%，其余类为 76.57%，重训为 77.64%。本题有部分遗忘的类，不能照抄“整类降到零”。最有用的是单独控制原类和错类：第 1 篇选谁，第 2 篇限定改哪里。第一轮只交换两个错类的分数，即使交换后没有收益，也可以排除“只是把分布调尖了”这个解释。[论文，第 4 节、表 1](https://arxiv.org/html/2503.23751v1)

### 3. Knowledge Distillation: A Good Teacher Is Patient and Consistent

**Beyer 等，CVPR 2022。** 图 3 直接比较了师生看不同随机裁剪、看相同随机裁剪等设置。用同一视图教学，学生能更好地复现教师函数。我们在遗忘侧给学生看裁剪或模糊图，代码却统一取原图的 `zt[idx]`。最好模型在原图和轻改图上的伪标签交叉熵均值分别为 5.2 和 4.1。差距给了实验入口，但真实重训模型看两版图也可以有不同输出。用上面第二轮的同视图对照，判断预测有没有跟着图像证据一起变。[论文，第 3.1.1 节](https://arxiv.org/pdf/2106.05237) · [对应的训练代码](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/xh_train.py#L144-L169)

### 4. Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time

**Wortsman 等，ICML 2022。** 从同一预训练点微调出的模型可以在权重空间求平均，发布时仍是一个模型。原文用验证集挑选加入平均的成员。队内已有一次直接证据：`r11` 为 0.2558、`r12` 为 0.2621，两者平均达到 **0.2658**。新的机会是把“只换错类”和“只换裁剪教师”两条同起点训练的权重各取一半，比较平均结果与两个单模型。现有最高分用的是带负系数的外推，论文检验的权重平均是另一种操作。[论文，第 3 节](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf) · [队内数据](https://github.com/sinonchum/euro-ai-championship-paris/blob/c73fedec70dc15c6422b9cffa51c6088a6fbe58e/task3/README.md#L68-L79)

### 5. Fine-Tuning is Fine, if Calibrated

**Mai 等，NeurIPS 2024。** 只用部分类别微调时，缺席类别整体的 logit 会偏移，类别内部的相对信息仍能保留。表 13 的 ImageNet-R/S 汇总里，通过平均 logit 差做校准，缺席类在全类别中分类的准确率从 3.5% 到 30.5%。本题的 55 类没有任何遗忘图。在剩余 91 类内部，`f7` 平均给这 55 类 **31.7%** 的条件概率，`t_world` 给 **19.3%**。可在小型重训中模拟一批缺席类，先判断偏差方向，再只调训练目标里 36 类与 55 类之间的总概率。原文做分布迁移下的分类，不能直接套用其偏移量或全局改正式模型的分类头。[论文，第 5 节、表 13](https://arxiv.org/html/2409.16223v1)

### 6. Knowledge Distillation as Semiparametric Inference

**Dao 等，ICLR 2021。** 作者用交叉拟合减少教师在自己训练样本上的过拟合，每折教师给没用过的那折打分，学生再用全部样本。我们已经有 `f7`；有价值的新实验在另外 2,331 张图上。对每个伪标签类别，把当前目标逐图的交叉熵由小到大排列，按 `f7` 估计的样本难度重新分给这些图，只调整原伪标签 logit。这样每类交叉熵的均值、标准差都能保持，能单独试“哪些图应当忘得更多”。现有折外训练只重置了后几层，浅层仍记得训练图。[论文，第 4.2 节](https://arxiv.org/pdf/2104.09732)

### 7. A Fast Knowledge Distillation Framework for Visual Recognition

**Shen、Xing，ECCV 2022。** FKD 把裁剪参数、翻转和该裁剪的教师预测一起缓存，让学生训练时重放同一输入。表 4 的一项 ResNet-50 对照中，准确率为 80.70% 对 80.67%，训练时间约为原来的 0.3 倍。我们保存的折外教师主要是原图预测。下次计算折外输出时，每图顺手存 4 个可重放视图，100 类完整 float32 概率合计只需 **5.6 MB**，另存变换参数。这能给第 3 篇提供真正的折外教师答案。训练仍保留在线采样的视图，避免记住少数固定裁剪。[论文，表 1、表 4](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840663.pdf)

### 8. Decoupled Knowledge Distillation

**Zhao 等，CVPR 2022。** 式 6 把完整 KL 写成“目标类对其余类”的二元 KL，加上非目标类条件 KL 乘以教师的非目标概率。在标准蒸馏里，如果教师对原类很自信，错类之间的监督自然就弱。表 1 的 CIFAR-100 实验中，保留监督交叉熵、只加非目标蒸馏可以超过普通 KD。我们可在已有解的训练中单独量出这两项；若两版已经匹配原类概率，学生仍学不准 91 类的排序，再只提高条件 KL 的训练权重。验证时仍用官方的完整 KL。[论文，式 6、表 1](https://arxiv.org/pdf/2203.08679)

### 9. What Makes Unlearning Hard and What to Do About It

**Zhao 等，NeurIPS 2024。** RUM 研究了遗忘和保留样本在特征上的重叠，以及逐样本记忆度。在 CIFAR-10 与 CIFAR-100 上，各记忆度组适合的遗忘方法并不相同。我们的类计数只能区分不同类的总体强度。可以用第 6 篇的折外误差在同一部分遗忘类中分组，专看高记忆度组是否从重分配中受益。论文使用真标签和保留数据，本题只能用伪标签和折外代理。队内另一版“按难度且高置信度”的目标只得 0.0089，因此要把难度排序与整体变尖分开试。[论文，第 4–5 节](https://proceedings.neurips.cc/paper_files/paper/2024/file/16e18fa3b3add076c30f2a2598f03031-Paper-Conference.pdf)

### 10. Ensemble Distribution Distillation

**Malinin、Mlodozeniec、Gales，ICLR 2020。** 同一个平均预测可以来自所有教师都不确定，也可以来自几个各自自信、意见相反的教师。原文通过教师预测分布建模区分两者。我们可直接用缓存算“平均概率的熵减去各教师熵的均值”，先只在种子意见一致的图上改目标。评分也必须区分两种平均：`f7` 的 5 个种子按官方顺序各算伪标签 CE 均值和标准差，再取平均，得到 **2.251 / 2.703**；先平均温度 2 的概率再还原 logits，则为 **2.021 / 2.447**。后者的平均概率只用于预测 KL。这里主要改善选模，2 个或 5 个种子也难以发现教师共同的偏差。[论文，第 3 节](https://arxiv.org/pdf/1905.00076) · [官方定义](Unlearning.pdf)

### 11. Unlearning’s Blind Spots: Over-Unlearning and Prototypical Relearning Attack

**Ha、Park、Yoon，ICML 2026。** 文中用 `OU@ε` 专测遗忘样本附近的保留样本受到多少连带损伤，并在邻近区域加入遮住遗忘类的蒸馏约束。对本题直接有用的是保留项。用小型实验真实的保留图，按冻结原模型的特征找最接近封顶类的那些图，检查加大分差后 `KL(orig || sub)` 是否主要在它们身上恶化。若是，只给相似的题内合成图加保留约束。正式比赛没有 control 图，不能凭代理估计是否已过 70%。[论文，第 3–4 节](https://proceedings.mlr.press/v306/ha26b.html)

### 12. Born-Again Neural Networks

**Furlanello 等，ICML 2018。** DKPP 实验保留教师最大预测，打乱其他类别对应的概率，用来分辨软标签里的收益有多少来自具体的类别身份。可把这个思路当第 1 名实验的判别手段：若教师选类和旧目标之间有差异，按相同交换次数、相同分数大小，在同伪标签的图片间随机换教师答案做对照。若随机交换与教师选择同样好，提升更应归因于训练扰动；只有教师选择胜出，才有理由扩大错类调整的范围。[论文，第 3 节](https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf)

### 13. Reference-Guided Machine Unlearning

**Mirlach、Laguna、Vogt，TMLR 2026。** ReGUn 从有标签的未见图片中，按遗忘批次的类别比例取参照，然后用批次平均预测蒸馏，同时在保留数据上训练。表 4 显示，在随机删除条件下，聚合后只剩逐点教师分歧的 2.6%–11.5%。本赛没有这种真实参照，且按图比较 KL。在我们的 346 张教师首选不同的封顶图上，拿同类均值来混一半，正向代理 KL 从 **0.486 变差到 0.527**，反向也从 **0.448 变差到 0.576**。因此只值得在另 36 类、折外估计最不稳定的图片上试小幅向同类均值靠拢。[论文，算法 1、表 4](https://arxiv.org/html/2603.11210v3)

### 14. Benchmarking Unlearning for Vision Transformers

**Zhao、Luca、Triantafillou，CoLLAs 2026。** 作者分别测试 ViT、Swin-T 和 DINOv2，直接对比基于留出重训的记忆估计和便宜的代理。文中的 Confidence 利用了训练过程中的预测，最终 checkpoint 的置信度无法补回这些信息。我们的原模型在遗忘图上已接近饱和。小型重训实验中可分别用折外难度和 `block 8` 置信度做第 6 篇的排序，看哪种在部分遗忘类上更接近重训的逐图损失。这篇提供选代理的依据，具体排序随架构和数据而变。[论文，第 4.1–4.3 节](https://arxiv.org/html/2602.20114v2)

### 15. Long-Tail Learning via Logit Adjustment

**Menon 等，ICLR 2021。** 原文根据类别先验调整 logit，推导的目标是各类等权的分类错误率。队内已经按遗忘图的每类张数分级，并试过用推测的剩余数量重加权。这里可借的是对假设做独立检验：在小型实验中保持同图的 91 类内排序不变，只看按已知剩余样本数调整组间概率能否改善 KL。正式数据中只知道删了多少张，不知道原来有多少张，特别是“每类原有 130 张”没有得到官方确认，所以排在基于逐图证据的方法之后。[论文，第 3 节](https://openreview.net/pdf?id=37nvvqkCo5)

### 16. An Information Theoretic Approach to Machine Unlearning

**Foster 等，TMLR 2025。** JiT 只需要遗忘图与原模型，用随机近邻扰动的有限差分降低输出对输入的敏感度。表 5 的 ViT-L 实验把遗忘集准确率从 100% 降至 10%，保留准确率也从 86.0% 降至 83.6%。我们只在微噪声这类几乎不改构图的变换上尝试轻量平滑，仍保留逐图教师目标；裁剪会改变可见证据，优先采用第 3 篇。单独的过度平滑会丢掉题目要求保留的类别差异。[论文，第 4 节、表 5](https://arxiv.org/html/2402.01401v4)

### 17. Localizing Memorization in SSL Vision Encoders

**Wang 等，NeurIPS 2024。** 在自监督 ViT 的第 12 个 block，作者报告 MLP 和注意力子层的增量记忆指标分别为 **0.022 和 0.010**。可以让当前方案只更新最后 4 个 block 的 MLP，比较在遗忘预测项达到相近值时，哪版对保留图的输出改动更少。自监督编码器的局部记忆测量与本题监督分类器有差异，论文数字只能帮助选择值得做的参数对照。[论文，表 2](https://proceedings.neurips.cc/paper_files/paper/2024/file/6f6af59b11f3919965b9811c6c9ad6df-Paper-Conference.pdf)

### 18. Patch Similarity Aware Data-Free Quantization for Vision Transformers

**Li 等，ECCV 2022。** PSAQ-ViT 从噪声出发，用 patch 相似度分布熵、类别目标和图像平滑项，合成适合 ViT 量化校准的样本。我们已有类别反演，可以只对没有遗忘图的 55 类加入 patch 相似度项，给保留侧提供另一种输入。对照用等量普通类别反演图，只看独立小型保留集上的 KL。作者验证的是量化精度，合成图在隐藏 control 上能保住多少能力仍须实测。[论文，第 3 节](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710154.pdf)

### 19. Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images

**Bonato、Cotogni、Sabetta，ECCV 2024。** SCAR 用收缩协方差的马氏距离找“最近的其他类”，再用分布外图像保留原模型行为。第 3.2 节需要原模型训练时保存的各类特征统计，并使用额外的分布外图片。我们缺 55 类的真实样本，因此直接在 45 个可见类里挑错类会系统性漏掉候选。只在有足够图的 36 类内部，可把折外特征距离作为两教师分歧时的第三个判断，再检查能否改善小型重训的条件 KL。[论文，第 3.2 节](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00004.pdf)

### 20. Remaining-data-free Machine Unlearning by Suppressing Sample Contribution

**Cheng 等，ICLR 2026。** MU-Mis 比较目标类和随机其他类 logit 对输入的梯度范数，调低二者之差，训练时可以完全不用保留集。连续子类删除实验中，末轮保留准确率为 84.59%，相应重训为 84.83%。它的优点是数据条件合适，代价是优化输入梯度需要额外的高阶反传，而且梯度强弱并不告诉学生应该选哪个其他类。只有当明确看到某组图原图已拟合、轻噪副本依然漏忘时，才值得在该组加一项低权重的敏感度损失。[论文，式 3、连续删除实验](https://arxiv.org/html/2402.15109v4)

## 复算教师间的 KL

在本仓库有对应分支历史的 clone 中，下面这段 NumPy 代码可以复现上表。每个条件分布先逐图归一化，再按 `KL(参照 || 估计)` 求平均。代码只读 Git 中已有的两份缓存。

```python
import io
import subprocess
import numpy as np

revision = "00a7a8d26f6d093bdc66a1ae90c970d6957cf90b"

def read_teacher(name):
    path = f"{revision}:task3_claude/gold_standin_{name}.npz"
    return np.load(io.BytesIO(subprocess.check_output(["git", "show", path])))

world, f7 = read_teacher("world"), read_teacher("f7")
labels = world["label"]
selected = world["capped_image"]
allowed = np.bincount(labels, minlength=100) < 129
group = labels[selected]

def conditional(teacher):
    values = teacher["p_T2_mean"][selected][:, allowed].astype("float64")
    return values / values.sum(axis=1, keepdims=True)

def mean_kl(ref, estimate):
    return (ref * np.log(ref / estimate)).sum(axis=1).mean()

def compare(source, reference):
    by_class = np.empty_like(source)
    for label in np.unique(group):
        rows = group == label
        by_class[rows] = (source[rows].sum(axis=0) - source[rows]) / (rows.sum() - 1)
    estimates = (by_class, (source + by_class) / 2, source)
    return [round(mean_kl(reference, item), 4) for item in estimates]

f, w = conditional(f7), conditional(world)
print(compare(f, w), compare(w, f))
print("共同首选", (f.argmax(axis=1) == w.argmax(axis=1)).sum(), "/", len(group))
```
