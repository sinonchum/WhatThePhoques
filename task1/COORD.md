# COORD.md：协调人给 Codex 的指令

最新的指令在最上面，带巴黎时间。每做完一步，读最上面一条并照做。
本文件、`RESULTS.md`、`fuse.py` 由协调人维护，对你只读。

---

## 2026-11-21 20:06 · 指令 11：融合破 0.6800 反馈 + 复刻 qin7 核心方法做 StegaStamp 全量矩形搜索 + 软替换连续响应

1. **重大阶段战报**：你的 `codex_boxfusion_pixelseal`（+0.0067）和 `codex_boxfusion_trustmark`（+0.0236）已全部汇入协调人的纯干净融合池！当前纯干净 `synth3` + 你的门控搜索，仅按 dev OOF 选权重，102 张留出总分已冲上 **0.6800**（PixelSeal **0.6941** / TrustMark **0.6591** / StegaStamp **0.6869**）。公开榜第 1 名目前已升至 **0.7011**，我们接下来的目标是用干净融合冲破 **0.70+**。
2. **从 qin7 最新 GitHub 分支研究出的核心突破（已验证有效，供你吸收）**：
   - qin7 之所以把单模型推到 0.676–0.718，靠的不是盲特征，而是三招：**① 原生 PyTorch 精确移植 StegaStamp ONNX 解码器**（带 STN 与无 STN 双路，与 ONNX 误差 `<4e-5`，彻底摆脱 CPU ORT 瓶颈并在 GPU 上跑出百倍吞吐）；**② 多尺度已知消息 `grad * act` 归因**；**③ 多视角局部裁剪（5-crop）STN 解码**——证明小块水印只要裁到局部框再进官方 STN，就能锁住坐标系读出极高的已知消息余量！
3. **你接下来的两项高价值任务**（请注意：**当前 JURECA SSH 登录暂时受限，暂不要发起新的 SSH 连接或重试，以免触发 JSC 封禁**；先在本机完成代码与软融合升级，待协调人通知 SSH 恢复后再投 `dc-gpu-devel` 跑 GPU）：
   - **任务 1（最高优先，补齐最后一块正交拼图）：StegaStamp 全量裁剪矩形搜索（`codex_cropsearch_stegastamp`）**
     - 目前干净版 StegaStamp V2 4-fold 的大面积已有 **0.6987**，但**小块只有 0.6653**（全队冲 0.70 的核心增长空间）。此前你在 6 张 dev 试跑 Stega 裁框拿过 **0.6064**，但当时因 CPU ONNX 太慢未全量跑。
     - 请在本机把官方 `models_codex_local/stegastamp/stega_stamp.onnx` 的解码器权重直接解析进原生 PyTorch 模块（STN + 7 层 Conv + FC，先在本机对 6 张图核验与 ORT max_abs_error < 1e-4），接入 `export_crop_search.py --method stegastamp`。若本机 MPS 批量推理可在 ~25 分钟内跑完 136/34/340，可直接本机跑并导出 `codex_cropsearch_stegastamp/{dev,holdout,test}.npz`（含 `box_score` 和 `diagnostics`）；若本机超过 45 分钟，则先跑完 136 `dev` 验证对干净 Stega 网络（已拉取在 `output/fusion/_coord/pulled/qin7_s3_ss_v2_4fold_stegastamp`）的切框增益，准备好 `dc-gpu-devel` 四卡脚本等 SSH 恢复后 5 分钟秒出全量。
   - **任务 2（本机立即可做，约 10 分钟）：修复 `boxfusion` 在多模型 Logit 融合中的“二值硬截断”损失**
     - 协调人发现：你的 `codex_boxfusion_{pixelseal,trustmark}` 在切框图片上把 `response` 写成了二值 `0.0 / 1.0`。当它与 qin7 4-fold 模型按 `0.5 : 0.5` 做 logit 平均时，`clip(1e-4, 1-1e-4)` 产生的 `±9.21` 极端 logit 会无差别压死所有切框图，既救对了真框，也放大了少数误切框的惩罚；若不设硬 `0/1`，而是将 `signed_raw_margin` 通过在 `dev` 上校准的单调映射转为平滑高置信度概率（例如框内 `sigmoid(a * (margin - thr) + b)` 约 `0.85–0.98`，框外保留基网络低响应或平滑衰减），多模型融合在边界与不确定框上会更稳。
     - 请在本机只用 136 `dev`（配合 `output/fusion/_coord/pulled/qin7_s3_v3_ps_4fold_pixelseal` 与 `qin7_s3_v3_tm_4fold_trustmark`）做 5 折嵌套 CV 验证“软概率切框 vs 硬 0/1 切框”，若 dev 净涨则导出 `codex_softbox_{pixelseal,trustmark}`。

---

## 2026-11-21 19:26 · 指令 10：PixelSeal 切 `dc-gpu-devel` 秒起（3 个空闲节点）+ 审计反馈 + 面积/连通域后处理校准

1. **审计非常精准**：你提前查出的 `train_stega_s3_kfold.py` ImportError 和打包接口不匹配已在 `15902536` 日志中全部应验（StegaStamp Stage 1/2 直接崩溃退出），协调人马上转告用户修复。另外协调人已核对 `SynthDataset` 源码：它严格只按 `pool.json["covers"]` 下标索引 `covers.npy`，而 `synth3` 的 `pool.json` 经查 100% 都是非留出 `train` 图（0 张测试图），所以传旧 `covers.npy` 配合 `synth3` 池不会读到测试图或留出图。
2. **GPU 队列加速（不用等到 22:10）**：刚查 `sinfo`，`dc-gpu-devel` 分区目前有 **3 个完全空闲的 4 卡节点**（`jrc[0319-0320,0448]`，无需 `--reservation=cispahack`，单作业最长 `--time=01:55:00`）。请直接把 `15902539` 取消，改投 `--partition=dc-gpu-devel`（去掉 `--reservation=cispahack`），4 卡并行预计 **15 分钟内**就能跑完 PixelSeal 136/34/340 全量裁剪搜索！跑完后按你说的先在 `dev` 上验证对 `claude_s3_ens_512_256_pixelseal` 的切框增益，有正增益再评一次 `holdout` 并同步本地。
3. **顺手做一项高性价比的后处理验证（仅在 dev OOF 上测试）**：
   - 官方 `miou_image` 是逐图两类平均 IoU：如果真值是纯全黑或纯全白，预测里多出哪怕 1 个噪点像素，缺失类 IoU 从 1.0 暴跌到 0.0，整图得分直接腰斩（上限 0.5）；同理，小块图若预测出极小碎块且未命中真框，得分远低于预测全 0（~0.45）。
   - 等 PixelSeal 跑上卡后，请在本机写一个轻量的 **Post-processing 校准实验**（只读现有干净 OOF 响应图：`claude_s3_ens_512_256_pixelseal`、你的 `codex_boxfusion_trustmark`、`codex_stegacutmix_stegastamp`）：在 5 折嵌套 dev CV 上测试（a）极小预测面积 `< area_min` 置全 0、（b）极大预测面积 `> area_max` 置全 1、（c）剔除极小孤立连通域 / 填补大面积内部极小噪点洞。预先设定保留门槛：**5 折嵌套 dev OOF 净涨 >= +0.003** 才固化规则，不看 holdout 调参。

---

## 2026-11-21 19:12 · 指令 9（hermes-gemini 接手协调）：PixelSeal 矩形搜索全量 + 严格干净合规审计

你的 `codex_stegacutmix_stegastamp`（留出 0.5914）和带 `box_score` 的 `codex_cropsearch_trustmark` 已收到并核验通过。
**重要反馈**：刚用你的 `box_score` 和 `whole_image_margin` 在 TrustMark dev OOF 上做门控切换（`signed_raw_margin >= 2.0` 且 `whole_image_margin < 0.3` 时换用你的矩形框），5 折嵌套 CV 在 dev 上从 0.6274 涨到 **0.6684**，固定规则在留出集上从 0.6250 涨到 **0.6486**（小块从 0.6410 涨到 **0.6883**）。你的矩形搜索路线在融合里兑现了 +0.0236 的净增益！

用户已做最新决定：
1. **严格执行方案 A**：后续提交的所有模型，合成底图严禁包含任何测试图或任何水印的 102 张留出图（目前 `synth2` 全系违规，Claude 的 `synth3` 3 种水印池已全部建好且干净）。
2. StegaStamp 主力采用 qin7 架构（需在干净 `synth3/stegastamp` 上重训），你的 `codex_stegacutmix_stegastamp` 作为纯真实图融合补充与合规保底。
3. 提交出口统一收口到协调人复核并经用户批准后单点执行，你继续不接触提交接口。

接下来请按优先级推进以下 **2 项任务**（可并行，先出结果的先写 `CODEX_STATUS.md`）：

### 任务 1（最高优先，直接涨分）：PixelSeal 全量裁剪矩形搜索 + `box_score` 导出
- 背景：主干分割网络在 PixelSeal 矩形小块上仍是明显短板（小块漏检多）。你此前在 6 张 dev 矩形小块 pilot 上拿到了 **0.5415**（解码一致率 ~0.63），当时未全量跑；现在有 TrustMark 门控成功的先例，只要挑出高置信度局部框就能在融合中救回矩形小块。
- 做法：复用 `export_crop_search.py` 在 **PixelSeal** 的 136 dev / 34 holdout / 340 test 上全量运行（建议用 JURECA GPU 作业跑，约 6 秒/图，单卡 ~50 分钟或按 batch/多进程加速；不训练，完全干净合规）。
- 导出：`task1/output/fusion/codex_cropsearch_pixelseal/{dev,holdout,test}.npz`（含 `box_score`）以及 `diagnostics/{dev,holdout,test}_rows.json`（含 `signed_raw_margin`、`whole_image_margin`、`message_accuracy`、`selected_is_whole_image`）。阈值只在 136 dev 上选，holdout 只评一次。

### 任务 2（本机只读，约 15 分钟）：独立代码合规与 OOF 泄漏审计
在等待 JURECA 跑任务 1 期间，请通过 ssh **只读**审查 JURECA 上的两组脚本（不修改任何远端文件，不碰 Claude 禁读目录以外的运行作业）：
1. 审查 `/p/scratch/training2666/qin7/task1/` 下的 `train_stega_ext.py`、`train_stega_5fold.py`、`train_v3_attr.py`、`train_v4_tm_rm.py`、`assemble_kfold.py`：
   - 核实它们默认或实际传入的 `--pool` 是否全部指向含测试图的 `synth2`；如果切到 `runs/synth3/<method>`，是否还需要改 `--covers` 或其他参数才能做到 **100% 不含 test 图且不含 102 张 holdout 图**？
   - 核实你在上一条提到的 `ShapeBank` 分折泄漏疑点：当折验证集（val fold）的真实 mask 形状是否进入了同折训练的 `ShapeBank`？影响程度如何？
2. 审查你自己的 `codex_stegacutmix_stegastamp` 与 `codex_cropsearch_trustmark` 的 `dev.npz`：确认哪几份是严格无泄漏的真 OOF（协调人准备在 `fuse.py` 中对真 OOF 放开权重/门控拟合）。
把审计结论简明写进 `CODEX_STATUS.md`，供协调人安排 qin7 干净重训参数。

---

## 2026-11-21 19:00 · 指令 8：用户确认继续要求干净素材；协调工作交接

- 用户确认：之后提交的模型只能用不含测试图、不含 102 张留出图的素材训练，测试图只做推理。你目前的方法都没用合成数据，不受影响。
- 队友 qin7 已提交过一次（公开榜 0.68，本地留出 0.6864），那一版的 StegaStamp 用了含测试图底图的旧素材。你的 `codex_stegacutmix_stegastamp`（只用真实 dev 图）因此是有用的干净备选，请照常完成留出集一次评分和 test 导出。
- 目前最弱的是 TrustMark。你的矩形搜索与另一代理的网络逐图分数相关系数约 0，但并集和按预测面积切换都没涨分。请把每张图“最佳框的已知消息一致率（或 logit 余量）”作为单独字段 `box_score`（`[N]` float32）加进 `codex_cropsearch_trustmark` 的三个 npz，融合时用它做切换判据。
- 协调工作将交给另一个代理（hermes-gemini），交接文档在 `task1/HANDOFF.md`。`COORD.md` 和 `CODEX_STATUS.md` 的约定不变。

---

## 2026-11-21 18:13 · 指令 7：用户决定，测试图不得用于训练

你现在可以用 `embed` 造合成数据（指令 4），但请遵守：

- 合成训练的底图不得包含测试集图片，也不得包含任何水印的 102 张留出图。只用 dev 图或外部干净图。
- 测试集图片只用于推理导出 `test.npz`。

其余不变。你的 TrustMark 定位器（留出 0.4540）和矩形搜索试跑结果已收到并记入结果表。矩形搜索请在 TrustMark 上跑全量并按 D 节导出；StegaStamp 的搜索是否值得再改一轮由你判断；PixelSeal 的矩形搜索另一代理在看，你不用做。

---

## 2026-11-21 17:40 · 指令 6：指令 5 的两项并行跑（用户已同意）

指令 5 里的第 1 项（TrustMark 官方矩形定位器）和第 2 项（裁出矩形再解码的搜索）互不依赖，请同时推进，各提一个作业，不要等一项做完再开始另一项。

- 两项各自单独报分、单独导出响应图（目录名分开）。
- 先出分的那项先写进 `CODEX_STATUS.md`，不用等另一项。
- 同时在跑的作业不要超过 3 个；登录节点每人限 20 个进程。

其余按指令 5 和指令 4。

---

## 2026-11-21 17:37 · 指令 5：接下来的优先级（用户已同意）

先读下面的指令 4（放开合成数据限制 + 两条造法事实），再按这个顺序做：

1. **TrustMark 官方矩形定位器**（最优先，目标半小时内出分）。隔离安装 trustmark 0.9.1 以上，`TrustMark(loadBBoxDetector=True).localize(img, return_all=True)`；主办方用的变体未知，Q 和 P 都试，用已知消息的解码一致率在 dev 上判断哪个变体对。报 TrustMark 的大面积 / 小块 / 合计，按 D 节导出响应图。
2. **裁出矩形再解码的搜索**，针对三种水印里的矩形小块。多尺度扫描后逐边微调，打分用已知消息的 logit 余量。先在 dev 的小块图上验证，再决定是否全量跑。
3. 遮挡探测停掉。你报的 StegaStamp 0.2789 已收到，会记入结果表。

口径提醒：筛选和调参只用 136 张 dev；留出集每个方法只报一次；每步做完更新 `CODEX_STATUS.md` 并导出响应图。

另一代理的最新情况（它的日志中间数，未入表）：把官方提取器的多层特征和已知消息一起喂给小解码器、纯合成训练，PixelSeal 在 1000 步时留出集约 0.61。所以 PixelSeal 上请不要重复这个组合；你的方案如果能在它错的图上对，融合才有收益。

---

## 2026-11-21 17:36 · 指令 4：放开合成数据限制（用户决定），附另一代理实测的两条造法事实

**限制变更**：用户决定放开指令 1 C 节的第一条。你现在可以用官方 `embed` 生成合成训练数据。其余不变：不读另一代理的脚本和目录，留出集不参与任何选择，不提交排行榜。

**另一代理的实测事实**（它自述，协调人没有复核；用户同意转给你，避免重走弯路）：

1. 真实图里的 StegaStamp 水印是低通的，没有 `wm.embed` 在 512 上直接嵌入时带的高频成分。把合成流程改成「先把图缩到 256、嵌入、再把差值双线性放大加回原图」后，网络对合成图和真实图的反应一致（它排除了 JPEG、整图 2 倍放大、高分辨率嵌入后缩小这三种解释）。含义：依赖 512 分辨率上高频残差的特征，在真实 StegaStamp 图上不成立。
2. PixelSeal 和 TrustMark 的 `wm.embed` 内部本来就是在 256 上算残差再放大，合成流程与真实一致；但不看消息的分割网络在这两种上学不动：合成数据训练 8000 步损失停在 1.56（等同随机），加全精度和高通前端的对照 1000 步内同样没动。文献（PixelSeal 论文）报告按最终强度 0.2 从头训练学不到水印，需要先用强度 1.0 再衰减，强度可通过 `model.blender.scaling_w` 调。

它的结果供你对照：纯合成、不看消息的分割网络，留出集 PixelSeal 0.2882、TrustMark 0.2682、StegaStamp 0.3409（小块 0.6010，大面积 0.0808），102 张总分 0.2991。官方提取器读真实水印是稳的（整图比特一致率 0.97–1.00）。

另外两点提醒：
- 筛选实验请用 136 张 dev 做交叉验证，不要用 34 张留出图筛。每个格子只有 17 张，0.01–0.02 的差别在噪声内。
- 另一代理正在做「官方提取器多层特征 + 消息匹配图 + 小解码器，纯合成训练」（PixelSeal、TrustMark）和「按第 1 条修正造法后的盲网络」（StegaStamp）。请避开这两个具体组合，优先做它没做的：TrustMark 官方定位器、裁出矩形再解码的搜索、遮挡探测，或把这些探测图作为输入的网络。

---

## 2026-11-21 17:31 · 指令 3：协调人的文献调研里和你的路线有关的事实（供参考，不是新任务）

来源是协调人安排的文献和源码调研，没有在本题数据上跑过。标了「推断」的不是文献结论。完整报告：`reports/隐形水印定位前沿方法.md` 的第 3、4、5、6 项（第 1、2 项属于另一条路线，请不要照着做）。

- 你的 smallseg 响应图已收到，`fuse.py` 跑通，分数与你报的一致。
- 三种提取器都是全局解码：PixelSeal 是 ConvNeXt 后空间平均再接线性层（`pixelwise: False`），TrustMark 是 ResNet-50 全局池化后接全连接，StegaStamp 是空间变换加全连接、没有池化。前两种的 8×8 图是精确的逐位置读数，StegaStamp 没有这种图，只能用遮挡或梯度。
- TrustMark 官方仓库主干有 `TrustMark(loadBBoxDetector=True).localize(img, return_all=True)`，返回归一化矩形框；权重只有 Q、P 两个变体；PyPI 0.9.0 起才有，3 月修过一个框的 bug，建议 0.9.1 以上。训练时用的是单独嵌在矩形上的水印。本题效果没验证。
- TrustMark 官方有去水印函数 `remove_watermark`，可作为干净图估计。
- TrustMark 四个变体 Q/B/C/P 互相解不出对方的消息；宽高比大于 2 时 Q/B/C 只在中心正方形嵌入，P 总是只嵌中心正方形。
- 对已带水印的图重新嵌入会覆盖旧消息（PixelSeal、StegaStamp 有文献数据），可用作遮挡时的填充。
- 推断：在裁剪块上单独嵌入的水印，图样在裁剪块自己的坐标系里，所以要「裁出来再解码」才读得到；整图嵌入后抠取的水印在整图坐标系里，适合原地遮挡。
- 遮挡类方法的文献设置（RISE）：4000–8000 个随机掩码，7×7 网格，保留概率 0.5，随机平移；打分建议用 logit 余量而不是比特准确率（后一句是推断）。

其余指令不变。

---

## 2026-11-21 17:13 · 指令 2：小型分割网络的核对结果（只有事实，没有新任务）

协调人用你保存的 `output/small_segmenter/*/predictions.npz` 和官方 `miou_image` 重算，已记入 `RESULTS.md`：

| 水印 | 留出集 大面积 / 小块 / 合计 | dev（训练用过的 136 张）大面积 / 小块 / 合计 |
| --- | --- | --- |
| PixelSeal | 0.4697 / 0.4926 / 0.4811 | 0.4663 / 0.4966 / 0.4814 |
| TrustMark | 0.4735 / 0.4711 / 0.4723 | 0.4903 / 0.4701 / 0.4802 |

- 每张图只给一个标签的上限是 0.4509（PixelSeal）和 0.4519（TrustMark）。
- 留出集上 PixelSeal 34 张里 25 张、TrustMark 34 张里 14 张的预测是整张图同一个标签。
- dev 是训练用过的图，分数和留出集几乎相同。

指令 1 的 A–E 仍然有效：请在 `task1/CODEX_STATUS.md` 写下一步打算，并按 D 节导出响应图（含 test）。

---

## 2026-11-21 17:08 · 指令 1：回报方式、分数口径、分工、要保存的文件

### A. 回报方式

每做完一步（一个作业出了结果，或者准备换方向之前），在 `task1/CODEX_STATUS.md` 最上面加一条，写：

- 时间
- 这一步做了什么
- 分数（按 B 的口径）
- 脚本路径和完整运行命令
- 输出路径
- JURECA 上正在跑或排队的作业号
- 下一步打算

这个文件归你写，协调人只读它，不会再打断你。没验证的内容写「没验证」。

### B. 分数口径（不满足就不进 RESULTS.md）

- 留出集：沿用你的 `split.json`（每种水印 34 张，seed 2026）。这 34 张不参与训练、阈值、训练轮数、归一化、模型选择中的任何一项。
- 评分：官方 `wmloc/utils.py` 的 `miou_image`，逐图算再平均。
- 每个方法、每种水印报三个数：大面积（`full_watermark=True`，17 张）/ 小块（`False`，17 张）/ 合计（34 张）。
- 参照线：每张图只给一个标签的上限是 0.454（见 `RESULTS.md`）。高于它才算定位到了区域。

### C. 分工

- 你的路线：只用真实训练图（每种水印 136 张）和官方提取器给出的信号。不要用官方 `embed` 生成合成训练数据，那条路线归另一个代理。
- 两边的方案要保持独立，所以不要读 `task1/` 里不是你写的脚本：`wmtask/`、`train_seg.py`、`gen_synth.py`、`assemble.py`、`prep_covers.py`、`diag_pipeline.py`、`baselines.py`、`jureca/synth_all.sbatch`、`jureca/cmd.sbatch`。也不要读 JURECA 上的 `claude_task1/`。
- 现在的小型分割网络（PixelSeal、TrustMark）继续做完并报分。
- StegaStamp 目前两边都没有分数。小型分割网络出第一个结果后，请为 StegaStamp 独立出一个方案并报分。

### D. 为融合保存响应图

每个你认为值得保留的方法、每种水印各存一份，从小型分割网络开始。

目录：`task1/output/fusion/codex_<方法名>_<水印>/`，例如 `codex_smallseg_pixelseal/`。
里面三个文件 `dev.npz`、`holdout.npz`、`test.npz`，字段相同：

| 字段 | 形状和类型 | 含义 |
| --- | --- | --- |
| `image_names` | `[N]` 字符串 | 图片名，不带 `.png` |
| `response` | `[N,128,128]` float32 | 阈值前的连续值，越大越像有水印 |
| `threshold` | 标量 float | 只用 dev 选出；二值预测 = `response >= threshold` |
| `dev_in_sample` | 标量 bool | 模型用 dev 图训练或拟合过就写 True |

- dev = 该水印的 136 张，holdout = 34 张，test = 测试集里该水印的全部图片。
- 测试集只做推理。不看公开榜，不对着它调任何东西。
- 模型用 dev 图训练过的话，`dev.npz` 最好给交叉验证的 out-of-fold 预测；给不了就照实写 `dev_in_sample=True`，融合时协调人不会用它的 dev 分数来挑权重。

### E. 不要做的事

- 不向排行榜提交。提交要先由用户看过留出集分数并同意，而且只由一方执行。
- 不把 API key 写进任何文件、日志或命令行。
- 不改 `task1/RESULTS.md`、`task1/COORD.md`、`task1/fuse.py`。
