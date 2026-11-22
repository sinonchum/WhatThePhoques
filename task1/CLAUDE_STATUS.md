# Claude (c01cea) 状态 — 最新的在最上面

口径：留出集 = `wmtask.common.holdout_names`（每种水印 34 张，共 102），官方 `wmloc/utils.py`；检查点按开发集选（混真实图的版本取最后一步）；阈值只在开发集或 out-of-fold 开发集上选。
JURECA 根目录：`/p/scratch/training2666/ding7/claude_task1/`（下文相对路径以它为根）。代码：`task1_v11/`，本地 `task1/` 同步。

## 2026-11-21 19:00 — 干净素材上追加种子/第二组交叉验证（在跑）

- 我没有提交过排行榜。我拼的 `runs/sub_v1.pt` 已校验但不提交（协调人 18:55 通知：qin7 已提交过一版，公开榜 0.68）。
- 占位作业 15901644（jrc0387，4 卡，约 20:20 结束）里正在跑，全部用干净素材池 `runs/synth3/`（底图只含训练集非留出图，无测试图）：
  - `runs/seg/pixelseal_s3_512_seed1`、`_seed2`：PixelSeal 全分辨率主干，纯合成，4000 步。用于和 `pixelseal_s3_512` 多种子平均。
  - `runs/seg/trustmark_s3_512real_b_cv{0..3}`：TrustMark 全分辨率主干，混 30% 真实 dev，4 折，seed 1，6000 步。用于和第一组 4 折平均。
- 预计 19:30 前出结果，届时导出到 `runs/fusion/` 并在这里追加。

## 2026-11-21 18:50 — 干净素材（synth3）结果，全部已导出到 `runs/fusion/<目录>/{dev,holdout,test}.npz`

格式：留出合计（大面积/小块）；开发集。

| 目录 | 说明 | dev_in_sample | 留出 | 开发集 | 阈值 |
|---|---|---|---|---|---|
| claude_s3_ext1_pixelseal | 256，纯合成 | False | 0.5954 (0.6083/0.5825) | 0.5934 | 0.25 |
| claude_s3_512_pixelseal | 主干全分辨率，纯合成，检查点 2000 步 | False | 0.6501 (0.6007/0.6995) | 0.6425 | 0.45 |
| claude_s3_ens_512_256_pixelseal | 上两者响应图平均 | False | 0.6591 (0.6201/0.6982) | 0.6451 | 0.50 |
| claude_s3_ext1_trustmark | 256，纯合成 | False | 0.5345 (0.5243/0.5446) | 0.5421 | 0.15 |
| claude_s3_ext3real_trustmark | 256，混 30% 真实 dev | True | 0.5839 (0.5868/0.5810) | 不作数 | 0.50 |
| claude_s3_512real_trustmark | 全分辨率，混 30% 真实 dev | True | 0.6133 (0.6186/0.6081) | 不作数 | 0.35 |
| claude_s3_ens_512real_256real_trustmark | 上两者平均 | True | 0.6287 (0.6167/0.6408) | 不作数 | 0.50 |
| claude_s3_512real_cv_trustmark | 全分辨率混真实，4 折；dev 响应为 out-of-fold | False | 0.6250 (0.6090/0.6410) | 0.6274（OOF，n=136） | 0.75 |

## 旧素材（synth2，底图含测试图和其他水印的留出图）——只作对照，不要删

| 目录 | 说明 | dev_in_sample | 留出 |
|---|---|---|---|
| claude_ext1_pixelseal | 256，纯合成 | False | 0.6369 (0.6315/0.6422) |
| claude_ext3real_pixelseal | 256，混真实 | True | 0.6289 |
| （未导出）runs/seg/pixelseal_ext4_512 | 全分辨率，纯合成，检查点 4000 步 | False | 0.6669；dev 0.6671 |
| claude_ext1_trustmark | 256，纯合成 | False | 0.5363 |
| claude_ext3real_trustmark | 256，混真实 | True | 0.5970 |
| claude_ext4_512real_testcovers_trustmark | 全分辨率，混真实 | True | 0.6383 (0.6070/0.6697) |
| claude_ext1_stegastamp | 官方解码器特征 + 解码头，纯合成 | False | 0.5688 (0.6075/0.5301)；dev 0.5849 |
| claude_ext3real_stegastamp | 同上，混真实 | True | 0.5680 |

## 方法与脚本

- 结构：`wmtask/extnet.py`。`ExtSegNet`（PixelSeal / TrustMark）= 官方提取器的可训练拷贝取四级特征（`--enc-res 512` 时主干吃全分辨率）+ 冻结拷贝给出 8×8 的“逐位置 logit × 已知消息符号”图 + 自顶向下解码器；`OrtFeatSegNet`（StegaStamp）= onnxruntime 读官方解码器 5 层卷积特征（冻结）+ 解码头。
- 合成数据：`gen_synth.py` + `wmtask/synth.py`。三种构造：整图水印挖洞；在裁剪矩形上嵌入；从整图水印版本抠物体贴到干净图。StegaStamp 用 `--lowres`（缩到约 256 嵌入、放大差值），因为真实 StegaStamp 水印是低通的。
- 训练：`train_seg.py --arch ext [--enc-res 512] [--p-real 0.3] [--fold k --nfolds 4 --select last]`。
- 导出：`export_fusion.py`（`--run` 给多个则平均）、`oof_fusion.py`（交叉验证拼装）、`assemble_fusion.py`（按图片名拼提交文件）、`submit.py`（校验；加 `--yes-submit` 才上传，读环境变量 `HACKATHON_API_KEY`）。

## 已知弱点和失败的尝试

- 矩形小块几乎找不到（PixelSeal 256 版：水印类 IoU 0.13，合成与真实一致）；大面积图里的洞只找到约三分之一。
- TrustMark 合成与真实有差距（256 纯合成：合成大面积 0.71，真实大面积 0.42），混真实图能补 0.06–0.10；差在哪没验证。
- 失败：从零训练的盲网络（PixelSeal/TrustMark 学不动；StegaStamp 学会合成但不迁移，留出 0.34–0.38）；重嵌入匹配滤波（AUC≈0.5）；PixelSeal 矩形搜索（每张 20 秒以上，已停）；PixelSeal 9 窗口消息图（每秒 1.4 步，500 步无优势，已停）；TrustMark 主干低学习率（无改善）。
