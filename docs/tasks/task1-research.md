# Task 1：哪些论文结果能迁移到这场比赛？

**决策：不把 MaskWM/WAM 自家编码器上的 IoU 当作本题预期分数。** 本题的三种写入器已经固定，而测试时提供每张图的写入方法和消息。主攻已知消息的空间证据，用不看消息的图像残差作为互补信号。

## 一手来源和适用性

- [Watermark Anything，ICLR 2025](https://arxiv.org/abs/2411.07231)：同时训练写入器和提取器，逐位置预测存在性和比特。可借鉴局部消息监督；没有证明其权重可直接识别比赛的三种水印。
- [Mask Image Watermarking，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/d74fb06615baaf6349007da7521223aa-Paper-Conference.pdf)：掩码引导的局部训练。我们可以保留官方写入器，合成不同面积的正、负区域，只训练读图端。
- [What Breaks Local Watermarks?，AISec 2026](https://arxiv.org/html/2609.16832v1)：在该文的无额外变换设置，MaskWM 的水印区域 IoU 为 0.902；定位比较只覆盖 MaskWM 和 WAM，使用各自写入器，不含 StegaStamp，且不是本题逐图两类 mIoU。这不是本比赛的 SOTA。该文区分“能正确解码”和“能定位”，提示不应在整图解码接近随机时一律返回全零。
- [原版 StegaStamp](https://github.com/tancik/StegaStamp) 有独立的 detector 和分割掩码，场景是相机拍摄的印刷图像，作者提示它对尺寸敏感。不要与赛事 ONNX 包装器（只给出全图 logits）混淆。[Adobe TrustMark](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py) 也支持可选的框检测；可做矩形候选，不应把框直接当成带洞或不规则形状的最终掩码。

## 不用 GPU 的实测对照

`python3 task1/residual_probe.py /path/to/train.zip`（numpy、scipy、Pillow）。在每个 16×16 图块统计亮度、色差的多方向高频残差，再加邻域与全图统计；按水印方法拟合岭回归，阈值预设为 0.5，不用消息或官方模型。按 `method × full_watermark` 分层，以图名哈希固定划分，训练 408，留出 102，每种方法 34 张。`full_watermark` 不进模型。尚未按原图来源去重，本轮只作方向判断。

- 固定留出集的逐图两类 mIoU：PixelSeal **0.2581**，TrustMark **0.2789**，StegaStamp **0.4081**，总体 **0.3150**。同一批图上，全零 **0.2598**，全一 **0.2402**。这些不是公榜成绩，也不能证明模型识别的是水印而不是生成和编辑痕迹。
- 查询时公开榜 Task 1 第一名 `jesters` 显示 **0.50**，第二名 **0.45**；只代表当时公开子集。

## 我们的候选：消息热图 + 局部残差，两条证据分别验证

1. PixelSeal 和 TrustMark：整图生成已知消息的逐位置分数；用同方法其他训练图的等长错误消息生成对照，校准局部分数。再对少量重叠窗口计算并映射回原图，与仅整图比较。避免在每张图强制标准化成必有前景，尤其考虑几乎全幅的例子。
2. StegaStamp：从已有信号较强的残差分支出发，加上粗网格遮挡时正确消息得分的变化。三类均只训练小型分割头，不要求旧版 StegaStamp 的相机检测器成功运行。
3. 配对合成监督（假设，不是已测算法）：仅从训练侧已知干净区取底图，用官方写入器和随机消息得到配对带水印版本。在同一底图、同一掩码上分别保留掩码内或外的水印；两张样本内容近似、标签互补。另做全干净对照与非矩形掩码，避免学会“看见拼接边界等于水印”。留出集保持真实、不合成。
4. 固定训练预算，比较残差、消息热图、两者融合和错误消息四组。按方法及面积报告 mIoU；确认真实留出集提升后才评估是否上榜。本次未评测融合模型，亦未向排行榜提交任何文件。
