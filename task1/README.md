# PixelSeal / TrustMark 消息匹配热图

使用官方提取器的 `per_position_logits`，计算
`mean_k((2 * message[k] - 1) * logits[k, y, x])`，双线性放大到 128×128。
不训练网络，不使用测试集，不上传比赛提交。

`wmloc/` 原样来自[官方 starter_kit.zip](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/tree/main)。
使用 Python 3.11，本目录只安装此次两种方法需要的依赖；模型版本与官方 starter kit 一致。

```bash
pip install -r requirements.txt
python prepare_heatmap_models.py --models-dir models
python message_heatmaps.py \
  --data /path/to/release \
  --models-dir /path/to/models \
  --out output/message_heatmaps
```

`release` 是官方 `train.zip` 的解压目录，内部应有 `train/metadata.json`。

JURECA 上从含 `.venv`、`release`、`models` 的任务目录运行：

```bash
mkdir -p output
sbatch jureca/run_heatmaps.sh
```

每种方法 170 张图，按 `full_watermark` 和水印面积分层，固定留出 34 张，随机种子为 2026。
`full_watermark` 只用于划分，模型输入只有图片和已知消息。
阈值仅用另外 136 张选择，目标严格使用官方逐图、两个类别平均的 mIoU。
同时用训练部分选择一个整图判断阈值，作为不进行空间定位的对照。

输出目录中，每种方法包含：

- `report.json`：训练和验证分数、阈值、按区域大小统计的结果。
- `validation.csv`：每张验证图片的分数及对照分数。
- `split.json`：固定的训练和验证图片名单。
- `heatmaps.npz`：全部图片的原始热图、128×128 热图和真实标签。
- `validation/*.png`：原图、消息匹配热图、预测区域和真实区域。
- `overview.png`：6 张最差、3 张中间、3 张最好的验证样例。

热图颜色范围统一使用训练集的第 1 / 99 百分位，不对每张图分别调颜色。
粉色表示有水印。低分辨率特征放大后的热图仅是定位信号，不是已校准的像素概率。

## 本次实跑结果

JURECA 作业 `15901376`，每种方法 136 张校准、34 张验证；两种方法原始热图均为 8×8。

| 方法 | 消息匹配热图 mIoU | 整图判断 mIoU | 差值 |
| --- | ---: | ---: | ---: |
| PixelSeal | 0.4625 | 0.4452 | +0.0173 |
| TrustMark | 0.4682 | 0.4500 | +0.0182 |

PixelSeal 在小水印和小缺口两组没有优于整图判断；TrustMark 在小缺口组反而更差。
直接对低分辨率消息匹配热图取阈值，提供的空间定位能力有限。
原始数值和全部验证对比图保存在 `output/message_heatmaps/`。

本次远端运行目录：`/p/scratch/training2666/ding7/message_heatmaps`，环境、模型和数据保留在该目录。

## 小型分割网络

`train_small_segmenter.py` 为 PixelSeal、TrustMark 分别训练一个小型 U-Net。
输入为原始 512×512 RGB、高频残差以及缓存的消息匹配热图，输出为 128×128 分割概率。
网络从上次热图预测开始学习修正，损失使用 BCE 加逐图两个类别的 soft IoU。

直接读取上次 `split.json`，每种方法仍为 136 张训练、34 张验证。
固定训练 80 轮，不用验证集选择训练轮数或阈值；热图归一化和最终阈值都只用 136 张训练图计算。
训练增强仅做图片、热图、标签同步水平翻转。

```bash
python train_small_segmenter.py \
  --data /path/to/release \
  --heatmaps output/message_heatmaps \
  --out output/small_segmenter
```

JURECA：`sbatch jureca/run_small_segmenter.sh`。
结果保存在 `output/small_segmenter/`，包含分数报告、逐图比较、模型权重、预测概率和训练损失。
`overview.png` 左列为提升最少的 6 张，右列为提升最多的 6 张；每组依次为原图、旧热图预测、新网络预测和真实标签。

### 小型网络实跑结果

JURECA 作业 `15901526`；每个网络 1,260,290 个参数，固定训练 80 轮。

| 方法 | 原热图 mIoU | 小型网络 mIoU | 增加的百分点 |
| --- | ---: | ---: | ---: |
| PixelSeal | 0.4625 | 0.4811 | +1.87 |
| TrustMark | 0.4682 | 0.4723 | +0.41 |

PixelSeal 的中等面积组由 0.4266 提高到 0.4749，小水印和小缺口基本没有改善。
TrustMark 的小缺口组由 0.4564 提高到 0.4712，小水印和中等面积组略降。
本次网络只带来小幅改善，小区域定位仍未解决。
每种方法的 `model.pt` 保存模型权重、阈值和原始训练/验证名单，`report.json` 保存完整分数。
