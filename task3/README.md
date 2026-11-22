# Task 3 Unlearning：本机(Mac MPS)流程与结论

公开榜当前最高 **0.2756**(`ex1413`,11-22 08:33)。榜首约 0.48。全部提交记录见 [`submissions.jsonl`](submissions.jsonl)。

本目录的 `xh_*.py` 只用官方给的 `model.pt` 和 `forget_images.npz`,不依赖 Modal,不用外部数据。

## 怎么跑

```bash
cd task3
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision timm numpy huggingface_hub requests pillow
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('SprintML/unlearning', repo_type='dataset', local_dir='starter')"
```

API key 放在 `~/.hackathon_key`(权限 600),`xh_submit.py` 从那里读,不会打印。

所有脚本都加 `PYTORCH_ENABLE_MPS_FALLBACK=1`,在 `task3/` 目录下运行。CUDA 机器要把 `xh_train.py` / `xh_eval*.py` 里 `device_type="mps"` 改成 `"cuda"`。

| 步骤 | 命令 | 耗时(M5) | 产出 |
|---|---|---|---|
| 1 原模型诊断 | `xh_diag.py` | 12 分钟 | `runs/xh_cache.pt`:原模型 logits、各层读出、重构图视图统计 |
| 2 缓存 block 7 输出 | `xh_tokens.py` | 7 分钟 | `runs/tok/*.pt`(8 GB):评估用的固定视图 |
| 3 留出线性读出 | `xh_oof.py` | 1 分钟 | `runs/xh_oof.pt` |
| 4 造目标 | `xh_target3.py` → `xh_target5.py` → `xh_target6.py` | 秒级 | `runs/t_*.pt` |
| 5 训练 | `xh_train.py` | 900 步约 17 分钟 | `runs/<name>.pt` |
| 6 本地检查 | `xh_eval2.py`、`xh_gate.py`、`xh_alt.py` | 各 1–2 分钟 | 见下 |
| 7 提交 | `xh_submit.py runs/<name>.pt "备注"` | 冷却 10–20 分钟 | 追加到 `runs/submissions.jsonl` |

只训练 blocks 8–11 + norm + head,blocks 0–7 冻结。半精度 autocast,约 1 秒/步(batch 48 遗忘 + 48 保留)。

### 复现当前最好的一条线

```bash
P=".venv/bin/python"; export PYTORCH_ENABLE_MPS_FALLBACK=1
$P xh_diag.py && $P xh_tokens.py && $P xh_oof.py
$P xh_target3.py t_mixed 10 2.5          # 9 个封顶类:记住的类压到最优其他类以下 10;其余:2.5 x block-8 读出
$P xh_target5.py t_grad4 4 1.0           # 其余类按遗忘集中的张数分级
$P xh_train.py --name r7  --target runs/t_mixed.pt --steps 1300 --lr 1e-4 --w_anchor 0.3
$P xh_train.py --name r9  --target runs/t_mixed.pt --init runs/r7.pt  --steps 1000 --lr 6e-5 --w_anchor 0.3 --w_t1 2.0
$P xh_train.py --name r10 --target runs/t_mixed.pt --init runs/r9.pt  --steps 900 --lr 6e-5 --w_anchor 0.3 --w_t1 1.0 --gap_full 8
$P xh_train.py --name r11 --target runs/t_mixed.pt --init runs/r10.pt --steps 900 --lr 6e-5 --w_anchor 1.0 --w_t1 1.0 --gap_full 10
$P xh_train.py --name r13 --target runs/t_mixed.pt --init runs/r11.pt --steps 900 --lr 6e-5 --w_anchor 1.0 --w_t1 1.0 --gap_full 10 --aug2 1 --same_class 1
$P xh_train.py --name r14 --target runs/t_grad4.pt --init runs/r13.pt --steps 900 --lr 6e-5 --w_anchor 1.5 --w_t1 1.0 --gap_full 10 --aug2 1 --same_class 1
$P xh_interp.py ex1413 r14:1.6 r13:-0.6  # 0.2756
```

## 提交记录(公开榜 30% 划分)

| 时间 | 模型 | 公开分 | 做了什么 |
|---|---|---|---|
| 03:43 | r2 | 0.0503 | block-8 读出目标,保留锚点权重 2 |
| 03:54 | r2 外推 ×1.5 | 0.0967 | 权重沿同方向外推 |
| 04:05 | r2 外推 ×2.0 | 0.0832 | |
| 04:18 | r3 | 0.0832 | block-8 目标拟合更紧 |
| 04:34 | r4 | 0.0383 | block-9 读出原始 logits(准确率 0.79,很软) |
| 04:56 | r5 | 0.0500 | 原模型在一个重构图视图上的输出(自信,准确率 0.75) |
| 05:27 | r3,head ×1.8 | 0.1377 | 只放大分类头 |
| 05:37 | r3,head ×3.0 | **0.0000** | 太尖锐 |
| 06:00 | r7 | 0.1083 | 封顶 9 类单独压低 |
| 06:10 | r7,head ×1.5 | 0.1788 | |
| 06:20 | r7,head ×1.9 | 0.1747 | |
| 06:32 | r8,head ×2.0 | 0.0089 | 按难度排序、自信型(准确率 0.70) |
| 06:46 | r9,head ×1.3 | 0.1861 | 封顶类准确率压到 0 |
| 07:08 | r10 | 0.2509 | 封顶类加 margin 损失(损失约 9.7),不放大 |
| 07:17 | r10,head ×1.2 | 0.2454 | |
| 07:30 | r11 | 0.2558 | 保留锚点权重 1.0,gap 10 |
| 07:48 | r12 | 0.2621 | 其余类按张数分级 |
| 07:58 | r11 与 r12 平均 | 0.2658 | |
| 08:07 | r13 | 0.2659 | 同类拼接锚点 + 更广的改动类型 |
| 08:23 | r14 | 0.2718 | r13 + 分级目标 + 锚点权重 1.5 |
| 08:33 | ex1413 | **0.2756** | r14 + 0.6·(r14 − r13) |

队伍此前最高是队友的 `fit_b8` 0.1476。

## 从分数里读出来的结论

每条后面是依据。没有 gold,本地算不出官方分,以下都是从公开分反推的。

1. **9 个"封顶类"gold 基本没见过。** 遗忘集按原模型预测落在 45 个类上,其中 9 个类恰好 130 或 129 张(类号 `0, 6, 7, 8, 29, 30, 31, 33, 38`)。把这些图的记住标签压到损失约 10、其余不变,分数从 0.186 升到 0.251(r9 → r10)。
2. **gold 在遗忘图上的损失很大。** 只放大分类头(预测类别不变、损失均值和标准差变大),r3 从 0.083 升到 0.138。现在最好的模型整体损失是原图 5.2 / 5.0、轻度改动 4.1 / 4.5(均值 / 标准差)。
3. **gold 的平均预测分布是分散的,不能尖锐。** 头部放大 ×3 得 0;自信型目标(T=2 最高概率 0.9)得 0.009。损失要大,但靠"记住的类很低、其余质量摊开"来实现,不靠"自信地押一个错类"。
4. **整类压掉这 9 个类会触发 70% 门槛。** 云端那版 `surg_b26`(这 9 类 head.bias −26,其他不动)得 0.000。按结论 1 它本该有分,所以最可能是验证准确率跌破 70%。推论:原模型验证准确率大约在 70%–77%。**遗忘必须是图片级的,不能变成类级的。** 最终排名用另外 70% 的数据,门槛余量不明,这是最大的风险。
5. **保留锚点越强,分数和安全性一起变好。** r11、r13、r14 逐步加强锚点,公开分每次 +0.005 到 +0.01,本地"连带损伤"指标同步改善。
6. **其余 36 个类也是忘得多一些更好。** 按遗忘集中的张数分级(张数越多压得越狠)+0.006(r12)。

未证实、正在验证的:

- gold 从没见过封顶类,所以对任何图都不该预测这 9 个类。当前模型在封顶类的图上有 31% 的 T=2 概率落在"其他 8 个封顶类"上。`xh_target6.py` 把目标里这部分清零并按各类剩余训练量重新分配(`t_v2`)。r15 用它训练,写这份说明时分数还没出。
- 每类训练图是否都是 130 张(结论 6 和上一条的重加权都依赖这个假设)。

## 本地检查的三个脚本

- `xh_eval2.py name[*k]`:原图和轻度改动上的损失均值 / 标准差 / 准确率,按封顶类和其余分开;以及在三种保留代理视图上与原模型的 KL。`*k` 表示分类头放大 k 倍。
- `xh_gate.py name ...`:门槛的代理指标。在"同类两张图拼接"和重构图裁剪上,对记住标签的准确率。原模型是 0.512 / 0.755;r10 是 0.290 / 0.442;r14 是 0.385 / 0.562。这些视图取自遗忘图本身,会夸大损伤(所有提交都过了公开划分的门槛),只用来比较候选之间谁更安全。
- `xh_alt.py name ...`:19 种轻微改动(翻转、模糊、噪声、旋转、缩放、平移、亮度、对比度、饱和度、灰度、降采样、遮挡、JPEG)下的损失。r11 在强模糊、降对比度、放大裁剪上遗忘变弱,`--aug2 1` 是为此加的。

## 训练脚本的参数

`xh_train.py`:

- `--target`:目标文件,含 `zt`(每张图的目标 logits)和 `full`(是否封顶类)。
- 遗忘侧损失:`KL_T2(target‖sub) + w_t1 · KL_T1(target‖sub)`,视图是原图(1/3)或现算的轻度改动(2/3)。
- `--gap_full g --w_gap`:封顶类的图上,记住的类 logit 至少比最优其他类低 `g`(hinge)。光靠 KL 压不到损失 10。
- 保留侧损失:`w_anchor · (KL_T2(orig‖sub) + 0.5 · KL_T1(orig‖sub))`,视图是现算的重构图裁剪、mixup、cutmix;`--same_class 1` 再加同类拼接。参考永远是**原模型**的输出(`--init` 续训时也是)。
- `--aug2 1`:遗忘侧一半的视图换成更广的改动(模糊、重采样、旋转、对比度)。
- `--freeze_absent`:是否冻结未出现的 55 个类的 head 行(r15 起设为 0)。

## 踩过的坑

- 用缓存的固定视图训练会被背下来:训练过的裁剪 KL 0.30,没见过的 0.52。所以每步现算视图。
- `xh_submit.py` 的冷却按服务器返回的秒数等。冷却在 10–20 分钟之间变化。
- `--init` 续训时,保留参考曾误用成续训起点的输出,已修(见 `xh_train.py` 里 `frozen_tail` 的注释)。
- 同一时间只能跑一个训练,MPS 不能并行。

## 没做的

- 没有本地 gold 代理评测(云端那条线在 Modal 上做过迷你复刻,见 `origin/task23_sync` 的 `task3_sync/`)。
- 没有在隐藏的 70% 划分上验证门槛余量的办法。
- 没试过训练 blocks 8–11 以外的层。
