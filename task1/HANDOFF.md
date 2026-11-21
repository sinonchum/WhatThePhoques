# Task 1 协调工作交接（给 hermes-gemini）

写于 2026-11-21 18:58（巴黎时间），写的人是原协调会话（Claude）。截止时间：周日 2026-11-22 12:00 巴黎时间。

你接手的是**协调人**的工作，不是做题。目标是让几条并行路线的产出合起来比各自单干好：维护结果表、分配路线、安排互相审查、做融合、把决定权交给用户（ding7）。

## 1. 先读这几个文件

| 文件 | 内容 |
| --- | --- |
| `task1/RESULTS.md` | 共享结果表，含入表条件、每个方法的分数、参照线、审查发现、分工变更记录 |
| `task1/COORD.md` | 给 Codex 的指令，最新的在最上面，目前到指令 7 |
| `task1/CODEX_STATUS.md` | Codex 每步写的状态，最新的在最上面 |
| `task1/fuse.py` | 融合和复核脚本，文件头有用法 |
| `reports/隐形水印定位前沿方法.md` | 文献调研报告（预期效果都没在本题上验证） |

## 2. 现在的分数（同口径：102 张留出图，官方评分）

| 水印 | 当前最好 | 谁 | 留出集 大面积 / 小块 / 合计 |
| --- | --- | --- | --- |
| PixelSeal | ext1（纯合成） | Claude | 0.6315 / 0.6422 / 0.6369 |
| TrustMark | ext3real（混真实 dev 图） | Claude | 0.5896 / 0.6045 / 0.5970 |
| StegaStamp | stegaext_live | qin7 | 0.6672 / 0.6478 / 0.6575 |
| 102 张总分 | | | **0.6305** |

- 参照线：每张图只给一个标签的上限是 0.4538。高于它才算定位到了区域。
- 上面三行都由协调人拉取响应图用 `fuse.py` 重算过，与各自报的一致。
- **这三个版本的合成素材都含测试图底图**，见第 5 节的未决事项。

## 3. 排行榜（18:58 更新，来自 qin7 的回报，协调人没有复核）

- 已提交 1 次：qin7 的 `runs/submissions/sub_v1.pt`。公开榜 Task 1 **0.68**（并列第 3，总榜第 2），本地留出集总分 0.6864。各水印 0.6839 / 0.6457 / 0.7295，都是多模型加权 logit 融合，名单和权重见 `RESULTS.md` 的“提交”一节。
- 本地与公开榜相差不到 0.01，说明这套留出集可信。但入选名单和权重是参照留出集表现手工配的，0.6864 略偏乐观，后续改进请用 dev 交叉验证来选。
- **这一版含测试图底图**，与用户 18:13 “停用测试图”的决定不一致。干净素材 `synth3` 只有 PixelSeal 和 TrustMark，StegaStamp 的还没建。
- 第 2 节的 0.6305 是单模型各取一个的旧数，已被这一版超过。
- 公开榜只是测试集的 30%，最终排名看另外 70%。不要对着公开榜调参或挑模型。排行榜只保留每队最高分。

## 4. 四条路线和怎么联系

| 谁 | 在做什么 | 工作目录（JURECA） | 怎么联系 |
| --- | --- | --- | --- |
| Claude 主对话 | PixelSeal、TrustMark 的“提取器多层特征 + 已知消息 + 小解码器”网络，合成数据训练 | `/p/scratch/training2666/ding7/claude_task1/` | 原协调人用 Claude 桌面应用的会话间消息。你用不了的话，请让用户转达 |
| Codex | 只用真实图的方法：TrustMark 矩形搜索、StegaStamp 小模型 | `/p/scratch/training2666/ding7/message_heatmaps/`，也在用户 Mac 本机跑 | 把指令写在 `task1/COORD.md` 最上面，它每步做完会读；它的回报在 `task1/CODEX_STATUS.md` |
| qin7（你这边） | StegaStamp，最近也有 TrustMark 的版本 | `/p/scratch/training2666/qin7/task1/` | 你自己 |
| xuanhao | 另一版 StegaStamp（0.6014，但留出集划分不同，不可比） | GitHub `sinonchum/euro-ai-championship-paris` 分支 `task1_xuanhao` | 通过用户 |

18:57 时在跑的作业：ding7 的 15901644（Claude，到约 20:23）；qin7 的 15902185、15902284 在跑，15902399、15902406 在排队。

**18:59 补充**：Claude 已回报 `s3_` 系列的分数，见 `RESULTS.md` 的“干净素材上的版本”一节（Claude 自述，协调人仍未拉取复核）。干净素材的最好版本是 PixelSeal 0.6591、TrustMark 0.6250。Claude 另拼了一份没提交的 `claude_task1/runs/sub_v1.pt`，与 qin7 已提交的 `sub_v1.pt` 同名但不同。

### 还没核对的新产物（接手后第一件事）

- Claude 用**不含测试图和留出图**的新素材重训的版本，在 `claude_task1/runs/fusion/` 下名字带 `s3_` 的目录：`claude_s3_ext1_pixelseal`、`claude_s3_ext1_trustmark`、`claude_s3_ext3real_trustmark`、`claude_s3_512_pixelseal`、`claude_s3_512real_trustmark`、`claude_s3_ens_512_256_pixelseal`、`claude_s3_ens_512real_256real_trustmark`。Claude 的正式回报还没到，协调人没有拉取和复核。
- Claude 的 `claude_ext1_stegastamp`、`claude_ext3real_stegastamp`（旧素材，仅作对照）。
- qin7 的 `qin7_stegaext_ens4_stegastamp`、`qin7_stegaext_synth_stegastamp`、`qin7_stegaext_real25*`、`qin7_stegaext_real35_stegastamp`、`qin7_v2_ss_real20_stegastamp`、`qin7_v2_tm_real25_trustmark`。只有 `qin7_stegaext_live_stegastamp` 复核过。
- Codex 的 `task1/output/fusion/codex_stegacutmix_stegastamp/`：它在 dev 上选定的 StegaStamp 最终版（dev 交叉验证 0.5797），留出集分数在生成中。它没用合成数据，不受测试图问题影响。

## 5. 等用户决定的事

1. **提交用哪个 StegaStamp**。用户 18:20 定的是用 xuanhao 的 mask；之后 qin7 交了同口径且更高的版本。协调人建议改用 qin7 的，用户还没回答。
2. **“停用测试图”：用户 19:00 已确认继续要求。** 之后提交的模型只能用不含测试图的素材训练，测试图只做推理；同时不得用任何水印的 102 张留出图做底图。已提交的 `sub_v1.pt` 不符合。需要做的：qin7 为 StegaStamp 建干净素材并重训（`synth3` 里目前只有 PixelSeal 和 TrustMark；StegaStamp 的合成要用“缩到 256 嵌入再放大差值”的流程）；PixelSeal、TrustMark 用 Claude 的 `s3_` 系列；融合时不要再放入旧素材的模型。
3. **以后由谁执行提交**。原规则是只由一方执行（Claude 主对话，用 `task1/submit.py`），且先报留出集分数给用户、用户同意才交。现在 qin7 也交过了，需要用户重新指定一个人。
4. **GCP 算力**：用户有 2 万美元额度。官方 PDF 没写是否允许外部算力；协调人建议先问主办方，目前算力不是瓶颈。

## 6. 规则（用户定的，接手后继续执行）

**入表条件，缺一条就不记：**
- 留出集 = `wmtask.common.holdout_names`（每种水印 34 张，seed 2026，共 102 张）。Codex 的 `make_split` 和它逐张相同，已核对。
- 评分 = 官方 `task1/wmloc/utils.py` 的 `miou_image`，逐图再平均。
- 留出图不参与训练、阈值、检查点、超参数、模型选择中的任何一项。
- 每个方法、每种水印报大面积（`full_watermark=True`，17 张）/ 小块（17 张）/ 合计。

**响应图格式**（各方都按这个交）：`<谁>_<方法>_<水印>/{dev,holdout,test}.npz`，字段 `image_names`、`response`（`[N,128,128]` float32，阈值前）、`threshold`（只用 dev 选）、`dev_in_sample`。dev 136 张、holdout 34 张、test 340 张。

**其他：**
- 提交前先把留出集分数报给用户，用户明确同意才交，只由一方执行。
- API key 在环境变量 `HACKATHON_API_KEY`，不写进文件、不打印、不发给任何代理。原协调人在本机和 JURECA 的 `ding7` 登录环境里都没看到这个变量。
- JURECA 登录失败不要重试，告诉用户；JSC 会封频繁重连的来源。
- 不动 `unlearning/` 和 `model_equality/`。
- 协调人只写 `RESULTS.md`、`COORD.md`、`fuse.py` 和 `task1/output/fusion/_coord/`。别人的代码和目录只读。
- 传递事实和数字，不传某一方的结论；没验证的就写没验证。

## 7. 怎么复核一个新结果

本机（Mac，只需要 numpy）：

```bash
cd "/Users/wenhanding/european ai hackathon/task1"
G=output/fusion/_coord/gt
uv run --with numpy python fuse.py --meta $G/train_metadata.json --masks $G/masks_128.npz \
  --src stegastamp=<目录1>,<目录2> --out output/fusion/_coord/<名字>
```

- 同一种水印的所有来源写在一个 `--src` 里，用逗号分隔；一种水印传两次会报错。
- 它按 dev 选来源，`dev_in_sample=True` 的来源不参与选择；留出集只用来报分。
- JURECA 上的响应图要先只读拉到 `output/fusion/_coord/pulled/`，例如：
  `ssh -o BatchMode=yes jureca 'cd <fusion 目录> && tar cf - <子目录>' > x.tar`（命令里不要夹带别的输出）。
- 真值副本在 `output/fusion/_coord/gt/`。

## 8. 已经知道的事实（省得重走）

**失败的方向（留出集分数）：**
- 提取器 8×8 逐位置 logits 取阈值：0.4625 / 0.4682（PixelSeal / TrustMark）。
- 只用 136 张真实图的小 U-Net：0.4811 / 0.4723，连训练图都没拟合。
- 不看消息的合成训练网络（Claude v1）：总分 0.2991。
- StegaStamp 固定模糊遮挡探测：0.2789。
- TrustMark 官方矩形定位器（0.9.2，Q 变体）：0.4540，没有定位效果。
- “已知消息重新嵌入再和高频做相关”：单元级 AUC 约 0.5。
- dev 里找同底图成对相减：一对也没有。

**数据的事实：**
- 三种造法：大面积 = 整图嵌入后把一块恢复成干净原图；小块约一半是在裁出的矩形上单独嵌入，另一半是从整图带水印的版本里抠出不规则物体贴到干净图上。
- 真实 StegaStamp 水印是低通的：合成时要先缩到 256 嵌入，再把差值放大加回，不能直接在 512 上嵌入。
- 主办方的 TrustMark 是 Q 变体（dev 整图已知消息一致率 Q 0.758，P 0.498）。
- PixelSeal 把正确的矩形裁出来读消息也只有约 0.61–0.69 的一致率；TrustMark 是 0.94–0.98；StegaStamp 0.81。

**见效的方向：** 把官方提取器的多层特征和已知消息一起喂给小解码器，用合成数据训练。TrustMark 混入真实 dev 图有用（+0.06），PixelSeal 没用。

**已知弱点（Claude 对 PixelSeal 的诊断）：** 矩形小块几乎找不到（水印类 IoU 0.13）；大面积里的洞只找到三分之一。

**融合：**
- TrustMark 上 Claude 的网络和 Codex 的矩形搜索（留出 0.5127，小块 0.5918）错的图基本不重合，“每张图取更好的一方”的不可达上限是 0.645。但直接取并集（0.580）和按网络预测面积切换（0.591）都不如单用网络（0.597）。缺一个可靠的切换判据。协调人想让 Codex 把每张图“最佳框的读消息一致率”单独导出来再试，**还没写进 COORD.md**。
- StegaStamp 上 qin7 的版本与 Codex 的 `stegacues` 简单平均是 0.6291，不如单用 qin7 的 0.6575。
- PixelSeal 取 ext1 而不是 ext3real 是看着留出集定的，两者只差 0.008，在噪声内，不要当成结论。

**独立审查（针对 Claude 的 ext 系列，Claude 已承认四条属实并在修）：** 合成底图含其他水印的留出图；含测试图；一个底图筛选阈值用了留出图的标签；混真实图的版本在训练见过的 dev 上选检查点和阈值。评分、检查点自动选择、导出划分没有发现错误。qin7 的脚本**没有审查过**。

## 9. 建议的接手顺序

1. 向用户和 qin7 要第 3 节的四样东西，更新 `RESULTS.md` 的提交记录。
2. 拉取并复核第 4 节“还没核对的新产物”，尤其是 Claude 的 `s3_` 系列，和旧分数并排记入表里。注意新素材底图从 760 张减到 204 张，分数下降可能来自多样性减少，不一定是之前虚高。
3. 把第 5 节的三个决定交给用户。
4. 用干净素材的版本拼出一份可提交的文件，把 102 张留出总分报给用户。拼测试集 mask 时按 `image_name` 取，不要按下标：测试集 metadata 里三种水印是交错的。
5. 之后再看融合增益：TrustMark 的矩形小块、PixelSeal 的两个弱点。

## 10. 原协调人没做完或做错过的

- `RESULTS.md` 和 `COORD.md` 里早期几个时间戳曾写得比实际快，已改正。
- 一次融合试跑里把 PixelSeal 传了两次，生成过一份人数和均分错误的报告（`output/fusion/_coord/fuse_B/`），那个数没有报给用户也没入表，脚本已加检查。
- xuanhao 的训练是否用过测试图：没查。
- 提交次数或冷却限制：任务说明 PDF 里没写，没验证。
