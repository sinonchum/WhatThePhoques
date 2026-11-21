# European AI Championship Paris

Team working repo for 21-22 November 2026.

## Briefs

- [Opening slides](docs/opening-slides-zh.md), deck: [EuroAIChampionshipParis2026-11-21.pdf](docs/EuroAIChampionshipParis2026-11-21.pdf)
- [Task 1: localize image watermark](docs/tasks/task1-zh.md), [Watermark_Localization.pdf](docs/tasks/Watermark_Localization.pdf)
- [Task 2: model equality](docs/tasks/task2-zh.md), [Model_Equality_Testing.pdf](docs/tasks/Model_Equality_Testing.pdf)
- [Task 2 可运行的分组流水线、离线演示和练习评分](task2/README.md)
- [Task 3: unlearning](docs/tasks/task3-zh.md), [Unlearning.pdf](docs/tasks/Unlearning.pdf)

Task 2's PDF was not in the latest upload. It is the copy already checked against the official brief.

## Rules that change the work

- Hacking ends Sunday 12:00 Paris time. Top 6 teams present for 6 minutes at 13:00.
- Task 1: 512x512 images, score on a 128x128 grid, mean IoU. Submit uint8 or bool masks. float32 exceeds the 20 MB limit.
- Task 2: watermark, system prompt, and decoding strategy do not change model identity. Family is 10%, model identity is 90%. Do not tune to the public score.
- Task 3: match the retrained model. No labels and no retain set. Below 70% validation accuracy both scores are zero.

Juelich: account `training2666`, partition `dc-gpu`, reservation `cispahack`. Do not use the backup project id `training2614`.
