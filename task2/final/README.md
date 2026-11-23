# Task 2 最终提交

公开分 **1.00**。方法和逐端点证据在 [experiments/fingerprint/README.md](../experiments/fingerprint/README.md)。

| 文件 | 内容 |
|---|---|
| [submission.csv](submission.csv) | 榜单记录：`t11`（G1 同模型组含 ep-12） |
| [submission-last-upload.csv](submission-last-upload.csv) | 最后一次上传：`t14`（ep-12 单独成模型），公开分同样 1.00 |
| [endpoints.csv](endpoints.csv) | 33 行：家族、同组或独立、信号、公开分是否核实过 |
| [receipt.json](receipt.json) | t11 回执，submission_id 3402 |
| [receipt-last-upload.json](receipt-last-upload.json) | t14 回执，submission_id 3590 |
| [build_submission.py](build_submission.py) | 从 `endpoints.csv` 生成两份 CSV |

```bash
python3 task2/final/build_submission.py --check
python3 task2/final/build_submission.py --out /tmp/t11.csv
python3 task2/final/build_submission.py --own ep-12 --out /tmp/t14.csv
```

公开分只核实了 13 个端点之间的 22 个同家族对子。ep-12 是隐藏端点，并入 G1 同模型组没有被核实。
