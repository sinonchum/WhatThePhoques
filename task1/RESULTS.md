# Task 1 共享结果表

维护人：协调会话。两个代理都只读这张表，要改数字请把结果交给协调人。

**入表条件（缺一条就不记）：**
1. 留出集 = 每种水印 34/170、seed 2026（`wmtask.common.holdout_names`，与 `message_heatmaps.py::make_split` 逐图一致），共 102 张。
2. 评分 = 官方 `wmloc/utils.py` 的 `downsample_mask` + `miou_image`（逐图、两类平均）。
3. 阈值、超参数、模型选择都没有用这 102 张图。

大面积 = `full_watermark=True`（留出集平均水印占比约 0.90），小块 = `full_watermark=False`（约 0.10）。每种水印大面积 17 张、小块 17 张。

最后更新：2026-11-21 20:08（巴黎时间，hermes-gemini 拉取 qin7 干净 s3 四折产物与 Codex 裁框，纯干净融合达 **0.6800**）

## 提交与合规状态

- 已提交次数：1 次（qin7 执行 `sub_v1.pt`，公开榜 **0.6813**，当前第 1 名 MasterChickenHack 升至 **0.7011**；旧提交含 `synth2` 测试图底图，按用户决定不作为后续合规依据）。
- **最新用户决定（19:10）**：
  1. **严格执行方案 A**：后续提交严禁使用含测试图或 102 张留出图底图的模型（`synth2` 全系停用；Claude 的 `synth3` 三种水印池均已核实 100% 仅含非留出 `train` 图）。
  2. **StegaStamp 主力路线定为 qin7**（已切到干净池 `runs/synth3/stegastamp` 重训产出），Codex 纯真实图模型作融合补充。
  3. **提交收口**：统一由协调人（hermes-gemini）按严格 dev OOF 融合算分并报用户批准后，通过唯一指定出口（`task1/submit.py`）执行。

**当前 100% 干净合规最好（`output/fusion/_coord/clean_v3_check`，仅按 dev 选权重，官方重算）：**
- **PixelSeal**：**0.6941**（大 0.6660 / 小 0.7223，dev 0.6829）—— `qin7_s3_v3_ps_4fold` (0.50) + `claude_s3_ext1` (0.25) + `codex_boxfusion_pixelseal` (0.25) Logit 融合。
- **TrustMark**：**0.6591**（大 0.6249 / 小 0.6933，dev 0.6866）—— `qin7_s3_v3_tm_4fold` (0.50) + `codex_boxfusion_trustmark` (0.50) Logit 融合。
- **StegaStamp**：**0.6869**（大 0.7044 / 小 0.6694，dev 0.6822）—— `qin7_s3_ss_v2_4fold` (0.75) + `codex_stegacutmix_stegastamp` (0.25) Logit 融合（注：JURECA 上 `qin7_s3_ss_v1_4fold`、`v2_synth`、`v4_tmrm_fast_s3`、`v3_ps384_s3` 已出或正在收尾，待 SSH 恢复拉取后继续叠高）。
- **当前纯干净 102 张留出总分**：**0.6800**（已追平旧违规版公开榜分，且 PixelSeal 超旧版 +0.0102、TrustMark 超旧版 +0.0134）。

## 方法

| 方法 | 谁 | PixelSeal 大 / 小 / 合 | TrustMark 大 / 小 / 合 | StegaStamp 大 / 小 / 合 | 留出集总分 (102 张) | 脚本 | 核对状态 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 小型 U-Net（RGB + 高频残差 + 消息匹配热图，136 张真实图训练，固定 80 轮，阈值在 dev 上选） | Codex | 0.4697 / 0.4926 / **0.4811** | 0.4735 / 0.4711 / **0.4723** | 没做 | 无（只覆盖 2 种，68 张平均 0.4767） | `task1/train_small_segmenter.py`，JURECA 作业 15901526 | 协调人用它保存的 `predictions.npz` 和官方 `miou_image` 重算，与 `report.json` 一致。留出集没有参与选择（读过脚本） |
| stegaext_live：PyTorch 移植的官方 StegaStamp 解码器 + 已知消息多层 grad×act 归因 + 高通支路，训练含真实 dev 图（dev 分数不作数），阈值 0.20 | qin7 | 没做 | 没做 | 0.6672 / 0.6478 / **0.6575** | 无（只覆盖 1 种） | 脚本路径未提供；响应图在 `/p/scratch/training2666/qin7/task1/runs/fusion/qin7_stegaext_live_stegastamp/` | 协调人拉取响应图用 `fuse.py` 重算，一致。留出集名单与我们相同。口径为 qin7 自述，脚本没有复核。自述基于 `synth2/stegastamp` 素材池，该池底图含测试图和其他水印的留出图 |
| ext1：官方提取器多层特征 + 8×8 消息匹配图 + 小解码器，纯合成训练；检查点和阈值按 dev 选 | Claude | 0.6315 / 0.6422 / **0.6369** | 0.5097 / 0.5628 / **0.5363** | 在训 | 无（2 种，68 张平均 0.5866） | `task1/train_seg.py --arch ext`、`wmtask/extnet.py`、`export_fusion.py`，作业 15901644 | 协调人拉取响应图用 `fuse.py` 重算，与 Claude 报的一致。脚本没有复核，待 Codex 审查 |
| ext3real：同上，训练混入 30% 真实 dev 图（dev 分数不作数） | Claude | 0.6322 / 0.6255 / **0.6289** | 0.5896 / 0.6045 / **0.5970** | 在训 | 无（2 种，68 张平均 0.6130） | 同上，加 `--p-real 0.3` | 同上 |
| 合成数据分割网络 v1（纯合成、从零训练、看不到消息；检查点按 dev 选，阈值固定 0.5） | Claude | 0.2851 / 0.2914 / **0.2882** | 0.2236 / 0.3128 / **0.2682** | 0.0808 / 0.6010 / **0.3409** | **0.2991** | `task1/train_seg.py` 等，JURECA 作业 15901535 | Claude 自述，协调人没有复核。Claude 判定失败，不导出融合文件 |
| StegaStamp 遮挡探测（16×16 图块模糊后看已知消息 logit 余量下降，不训练，阈值在 dev 上选） | Codex | 没做 | 没做 | 0.1188 / 0.4391 / **0.2789** | 无（只覆盖 1 种） | `task1/stegastamp_occlusion.py`，JURECA 作业 15901640 | 协调人用它导出的响应图和 `fuse.py` 重算（见下方日志）。Codex 判定失败，已停 |
| 裁出矩形再解码的搜索（多尺度粗扫 + 逐边微调，不训练，阈值在 dev 上选） | Codex | 没做 | 0.4336 / 0.5918 / **0.5127** | 试跑未全量 | 无（只覆盖 1 种） | `task1/crop_search.py`，JURECA 作业 15901913 | 协调人用它导出的响应图和 `fuse.py` 重算 |
| TrustMark 官方矩形定位器（0.9.2，Q 变体）+ 已知消息验证候选框，不训练，阈值在 dev 上选 | Codex | 没做 | 0.4583 / 0.4497 / **0.4540** | 没做 | 无（只覆盖 1 种） | `task1/trustmark_localizer.py`，JURECA 作业 15901818 | 协调人用它导出的响应图和 `fuse.py` 重算。等于整图标签上限，没有定位效果 |
| 消息匹配热图（8×8 逐位置 logits，训练部分定阈值） | Codex | 0.4723 / 0.4526 / **0.4625** | 0.4496 / 0.4868 / **0.4682** | 没做 | 无（只覆盖 2 种，68 张平均 0.4653） | `task1/message_heatmaps.py`，JURECA 作业 15901376 | 协调人用它保存的 `heatmaps.npz` 和官方 `miou_image` 重算，与 `report.json` 一致 |

## 参照线（不是方法，协调人用官方评分在同一留出集上算的）

| 参照 | PixelSeal 大 / 小 / 合 | TrustMark 大 / 小 / 合 | StegaStamp 大 / 小 / 合 | 留出集总分 |
| --- | --- | --- | --- | ---: |
| 全部预测为无水印 | 0.0511 / 0.4528 / 0.2520 | 0.0459 / 0.4497 / 0.2478 | 0.0432 / 0.4607 / 0.2520 | 0.2506 |
| 全部预测为有水印 | 0.4489 / 0.0472 / 0.2480 | 0.4541 / 0.0503 / 0.2522 | 0.4568 / 0.0393 / 0.2480 | 0.2494 |
| 每张图只给一个标签，且标签选对（整图判断的上限） | 0.4489 / 0.4528 / 0.4509 | 0.4541 / 0.4497 / 0.4519 | 0.4568 / 0.4607 / 0.4587 | 0.4538 |

读法：任何不做空间定位的方法最高只能到约 0.454。分数要明显高于 0.454 才说明真的定位到了区域。

小型 U-Net 的两个附带事实（协调人从 `predictions.npz` 算的）：
- 在它训练用过的 136 张 dev 图上也只有 0.4814（PixelSeal）和 0.4802（TrustMark），和留出集几乎一样。也就是说它连训练图都没有拟合出区域。
- 留出集上 PixelSeal 34 张里有 25 张、TrustMark 34 张里有 14 张的预测是整张图同一个标签。

补充核对：Codex 自己写的阈值搜索评分（`threshold_scores`）在 136 张 dev 上给出 0.4664 / 0.4646，协调人用官方 `miou_image` 在同样的图和阈值上重算得到相同的数。

留出集名单核对：把 Codex 的 `make_split` 原样取出来在三种水印上运行，得到的 34 张与 `wmtask.common.holdout_names` 逐张相同（PixelSeal、TrustMark、StegaStamp 都是）。Codex 之后做 StegaStamp 只要沿用 `make_split(seed=2026)`，就和 Claude 的留出集一致。

Claude 独立算的三条参照线（全 0、全 1、整图标签上限）在留出集上与本表逐项相同。

## 留出集被接触过的地方（如实记录，供互相审查时判断）

- Claude 的诊断脚本 `task1/diag_pipeline.py`（作业 15901460）统计了全部 510 张训练图，含留出的 102 张。它据此判断数据的构造方式，并在设计合成数据时用了一个全集统计数（小块中矩形占比 0.51）。合成时实际用的形状库只读 dev 的 mask。来源：Claude 自述，协调人没有复核代码。
- Claude 的训练每 500 步在该水印的 170 张真实图上评一次，dev 和留出集分开打印；挑检查点只看 dev，阈值固定 0.5。来源：Claude 自述，没有复核。
- Codex 的消息匹配热图：阈值只在 136 张 dev 上选。协调人读过脚本，属实。

## 待入表（在跑，还没有正式分数）

| 方法 | 谁 | 覆盖 | 状态 |
| --- | --- | --- | --- |
| stegastamp_ext1 / ext3_real：ONNX 解码器 5 层特征 + 消息 logit，接可训练解码头 | Claude | StegaStamp | 作业 15901644 内运行。Claude 报 500 步时留出 0.5165 / 0.5025（中间数） |
| 裁出矩形再解码的搜索，全量 | Codex | TrustMark 为主 | 试跑完成，全量未跑 |

## 干净素材（synth3，不含测试图和留出图）上的版本（19:15 协调人已拉取并用 `fuse.py` 逐项重算核实）

导出目录都在 `claude_task1/runs/fusion/`，本地副本已拉到 `output/fusion/_coord/pulled/`。留出集 合计（大面积 / 小块），与 Claude 自述完全一致：

| 目录 | 说明 | 留出集 | dev | 合规状态 |
| --- | --- | --- | --- | --- |
| `claude_s3_ext1_pixelseal` | 256，纯合成 | 0.5954 (0.6083 / 0.5825) | 0.5934 | 干净 |
| `claude_s3_512_pixelseal` | 全分辨率主干，纯合成 | 0.6501 (0.6007 / 0.6995) | 0.6425 | 干净 |
| `claude_s3_ens_512_256_pixelseal` | 上两者响应图平均 | **0.6591** (0.6201 / 0.6982) | 0.6451 | 干净 |
| `claude_s3_ext1_trustmark` | 256，纯合成 | 0.5345 (0.5243 / 0.5446) | 0.5421 | 干净 |
| `claude_s3_ext3real_trustmark` | 256，混 30% 真实 dev | 0.5839 (0.5868 / 0.5810) | 训练见过 | 干净（无 OOF） |
| `claude_s3_512real_trustmark` | 全分辨率，混真实 | 0.6133 (0.6186 / 0.6081) | 训练见过 | 干净（无 OOF） |
| `claude_s3_ens_512real_256real_trustmark` | 上两者平均 | 0.6287 (0.6167 / 0.6408) | 训练见过 | 干净（无 OOF） |
| `claude_s3_512real_cv_trustmark` | 全分辨率混真实，4 折交叉验证，留出和测试取 4 折平均 | **0.6250** (0.6090 / 0.6410) | OOF 0.6274 | 干净 + 真 OOF |
| `codex_stegacutmix_stegastamp` | Codex 87 通道低通+CutMix，仅 136 真实 dev | **0.5914** (0.6506 / 0.5322) | OOF 0.5797 | 干净 + 真 OOF |

### qin7 与旧 `synth2` 系列全量复核（含测试图底图，仅作架构实力对照，不入干净提交）
- **PixelSeal**：`qin7_v2_ps_synth` 留出 **0.6678**（0.6248 / 0.7108，dev 0.6639）；`qin7_v2_ps_real25` 留出 **0.6719**（0.6366 / 0.7072）。
- **TrustMark**：`qin7_v3_tm_4fold` 留出 **0.6346**（0.6177 / 0.6515，dev OOF 0.6582）；`claude_ext4_512real_testcovers` 留出 **0.6383**。
- **StegaStamp**：`qin7_stegaext_synth` 留出 **0.6838**（dev 0.6869）；`qin7_v2_ss_real20` **0.7014**；`qin7_stegaext_real35` **0.7121**；`qin7_stegaext_real25` **0.7185**；`qin7_stegaext_ens4` **0.7257**（0.7054 / 0.7459）；Claude 的 `ext1_stegastamp` 仅 0.5688。
- **关键发现**：JURECA 上 `claude_task1/runs/synth3/stegastamp`（272 张非留出 train 底图，0 测试图）已于 19:08 构建完毕！但 qin7 的所有脚本（含正在跑的 `v3_ps4f`、`v4_tmrm`）默认 `--pool` 仍指向 `runs/synth2/<method>`。只需把 qin7 的 StegaStamp / PixelSeal / TrustMark 4-fold 脚本加上 `--pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/<method>` 重训，即可拿到全套 0.67+ / 0.65+ / 0.72+ 的 100% 干净模型。

### 融合突破（19:12 协调人实测验证）
- **TrustMark 门控切换成功兑现互补收益**：利用 Codex 新增的 `box_score`（`signed_raw_margin`）与整图 `whole_image_margin`，当局部框 `signed_raw_margin >= 2.0` 且 `whole_image_margin < 0.3`（整图读不出消息但局部框强匹配）时切换为 Codex 搜索框。
  - 在 `claude_s3_512real_cv_trustmark` 上做严格 5 折嵌套 CV：dev OOF 从 0.6274 升至 **0.6684**（小块从 0.6371 升至 **0.7191**）；固定规则在留出集上从 0.6250 升至 **0.6486**（小块从 0.6410 升至 **0.6883**，净涨 **+0.0236**）。
  - 同规则配 `claude_s3_ens_512real_256real_trustmark`：留出集达 **0.6580**（大 0.6167 / 小 0.6993）。

- 新旧素材对照（Claude 自述）：PixelSeal 256 纯合成 0.6369 → 0.5954；PixelSeal 全分辨率 0.6669 → 0.6501；TrustMark 256 纯合成 0.5363 → 0.5345；TrustMark 256 混真实 0.5970 → 0.5839；TrustMark 全分辨率混真实 0.6383 → 0.6133。新素材低 0.00–0.04，底图从 760 张减到 204 张和去掉泄漏两个因素分不开，差别在噪声量级。
- Claude 拼好但**没有提交**的文件：`claude_task1/runs/sub_v1.pt`（PixelSeal 用 `s3_ens_512_256`，TrustMark 用 `s3_512real_cv`，StegaStamp 用 xuanhao 的 mask），`submit.py` 校验通过。另有全用 Claude 模型的对照 `runs/sub_v1_allclaude.pt`（留出 102 张均值 0.6176）。注意它和 qin7 已提交的 `sub_v1.pt` 同名但不是同一个文件。
- 干净版本对比已提交版本的同水印分数：PixelSeal 0.6591 对 0.6839，TrustMark 0.6250 对 0.6457。

## 独立审查：Claude ext 系列（18:12，全新的只读 Codex 实例，只给了代码和数字）

审查原文：协调人 scratchpad 的 `codex_review_ext.out`。协调人没有逐条复核代码，下面是审查的结论。

没发现问题的：评分确实用官方 `miou_image`、输出 128×128；检查点自动选择只看 dev；`--p-real` 只取 dev 图；形状库排除了留出 mask；导出的划分和阈值与声明一致。

发现的问题：
1. 合成底图没有排除其他水印的留出图。PixelSeal、TrustMark 各 255 张 train 底图里有 51 张是其他水印的留出图（`wmtask/synth.py:54`）。同水印的留出图被排除了，所以留出图本身没有进对应模型。影响未定量。
2. 测试集图片被用作合成底图（`prep_covers.py:30`，其他水印的测试图）。只用了像素，没有用标签。是否符合比赛规则需要用户判断。
3. 底图筛选阈值的拟合用到了留出图的 `full_watermark` 标签（`wmtask/synth.py:35`）。影响间接。
4. 合成方案的设计参考过含留出集的全集诊断（上面已记录）。
5. ext3real 的检查点和阈值都在训练见过的 dev 上选，选不出泛化最好的阶段。
6. 结果表曾按留出分数给每种水印挑赢家（已改，见最上面）。

## 试跑和诊断（不是留出集分数，不入方法表）

- 17:56 Codex 的“裁出矩形再解码”搜索试跑（`task1/crop_search.py`，作业 15901819）：每种水印只取 6 张 dev 的矩形小块图。数字见 `output/crop_search_pilot/report.json`，协调人只读了报告，没有重算。
- 17:56 Codex 实测主办方用的 TrustMark 变体是 Q：136 张 dev 整图已知消息比特一致率 Q 为 0.758，P 为 0.498（等同随机），5 折结果一致。

- 18:33 qin7 澄清：之前的 0.469 是早期非同口径的本地数，作废；qin7 没有单独提交过公开榜。纯合成版和 5 折 OOF 版在跑。

## 第三条路线：队友 xuanhao 的 StegaStamp（18:18 核对，留出集划分不同，不进方法表）

来源：`github.com/sinonchum/euro-ai-championship-paris` 分支 `task1_xuanhao`（提交 e3a4cda，18:16），协调人下载到 scratchpad 只读查看。

- 方法（他的说明）：已知消息与相反消息的编码器残差 + 图像取证特征 + 局部统计，接一个 mask 头和一个极性头。在 Mac 上训练。
- 交付物：`task1/handoff/stegastamp-hold-v1.npz`（他的 34 张留出图的二值 mask）和 `stegastamp-test-v1.npz`（340 张测试图的二值 mask），以及合并脚本 `merge_stegastamp.py`。只有二值 mask，没有连续响应图。
- 协调人用官方 `miou_image` 在他的 34 张留出图上重算：大面积 0.6515 / 小块 0.5514 / 合计 **0.6014**，与他报的一致。34 张里有 6 张是整图单一标签。
- **划分不同**：他的留出集按文件名哈希取（`train_grid.py::split_ids`），和我们的不是同一批。他的 34 张里只有 8 张在我们的留出集里（这 8 张上是 0.5620），其余 26 张在我们的 dev 里；反过来，我们的 34 张 StegaStamp 留出图里有 26 张被他用于训练或选模型。所以他的模型不能在我们的留出集上评，这个 0.6014 不能和表里的数直接比。
- 他自己报的三种水印基线：他的 102 张留出图上 0.5526（PixelSeal 0.5204、TrustMark 0.4697、StegaStamp 0.5191 是更早一版）。没有公开榜分数。
- 是否用过测试图或我们的留出图做训练底图：没查。

## Claude 实测的失败原因（v1，Claude 自述，协调人没有复核）

- 真实图里的 StegaStamp 水印是低通的，没有 `wm.embed` 在 512 上现嵌时带的高频成分。把嵌入改成“缩到 256、嵌入、差值双线性放大加回”后，v1 网络的输出与真实图一致（0.03–0.05 对 0.052）。
- PixelSeal、TrustMark 的合成损失 8000 步停在 1.56，盲网络没学到东西；全精度加高通前端的对照 1000 步内同样没动。
- 官方提取器读真实水印是稳的（整图比特一致率 0.97–1.00）。

## 分工变更记录

- 19:00 用户确认：继续要求。之后提交的模型只能用不含测试图的素材训练（测试图只做推理）。已提交的 `sub_v1.pt`（公开榜 0.68）不符合这条，保留在榜上但不作为最终方案的依据。待办：qin7 为 StegaStamp 重建干净素材并重训；PixelSeal、TrustMark 用 Claude 的 `s3_` 系列。
- 18:20 用户决定：第一版提交的 StegaStamp 部分用队友 xuanhao 的测试集 mask（v1，副本在 `output/fusion/_coord/teammate_stegastamp_v1/`）。因此报给用户的 102 张总分会是拼出来的：PixelSeal、TrustMark 用我们的留出集，StegaStamp 用他的留出集（0.6014）。
- 18:13 用户决定：测试集图片不得再用作合成训练的底图（推理仍可）。已转给 Claude，并要求重建素材池时同时排除全部 102 张留出图。现有 ext 系列的分数都来自含测试图底图的素材池；最终提交必须用重建后的模型。
- 17:36 用户决定：放开 Codex 不得用 `embed` 造合成数据的限制，并把 Claude 实测的两条造法事实转给 Codex（见 `COORD.md` 指令 4）。此后两边的方案不再完全独立。

## 融合

脚本：`task1/fuse.py`。按水印类型在 dev 上挑单一来源或混合，在留出集上报分，写出测试集 mask。留出集不参与任何选择。

- 18:07 第一次真实融合（`output/fusion/_coord/fuse_A`）：PixelSeal 和 TrustMark 各放入 Claude ext1 与 Codex 的全部来源。按 dev 选出的都是 Claude ext1 单独使用，留出集 0.6369 / 0.5363，68 张平均 0.5866。Codex 现有来源没有带来增益。
- 18:24 TrustMark 上 Claude 网络与 Codex 矩形搜索的互补性：逐图分数相关系数约 0（留出集 0.18 / -0.02），“每张图取更好的一方”的不可达上限是 0.5948（配 ext1）和 0.6453（配 ext3real），其中小块 0.67–0.70。但两个实际可用的规则都没涨：直接取并集 0.5191 / 0.5796；按“网络预测的水印面积小于阈值就换用搜索框”（阈值在 dev 上选为 0.30，dev 从 0.5534 涨到 0.5882）在留出集上是 0.5345 / 0.5907，略低于单用网络。结论：互补潜力存在，但还没有能兑现它的规则。
- ext3real 的 dev 是训练见过的，脚本不会用它的 dev 分数做选择；是否采用它要靠交叉验证的 out-of-fold 预测，还没有。
- Claude 的响应图副本在 `output/fusion/_coord/pulled/`；真值缓存在 `output/fusion/_coord/gt/`。
