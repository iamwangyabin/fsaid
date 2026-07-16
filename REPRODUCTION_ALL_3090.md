# RTX 3090 全量可执行实验汇总

执行日期：2026-07-15 至 2026-07-16。

## 范围与结论

本轮在 `3090` 服务器上跑完了当前数据和官方权重允许执行的全部核心二分类实验：

- GenImage：FTNet/FTNet-T 论文 Table 2 公开协议、CLIPDet、OmniDFA、六生成器留一诊断；
- OpenSDI：FTNet/FTNet-T 论文 Table 4 公开协议、CLIPDet、OmniDFA、K=1/5/10 且三个随机种子的持续学习矩阵；
- SynthBuster：CLIPDet论文Table 4的四个商业生成器官方固定推理；
- 合计保存 298 条新固定权重/留一/持续学习记录，另有此前完成的 22 条论文协议记录。

最明确的论文数值复现是 OmniDFA：GenImage 官方 56,000 张聚合测试得到
`95.8571%` Accuracy，四舍五入为论文 Table 4 的 `95.86%`。

当前唯一无法有效运行的已接入方法是 FSD。
[官方仓库](https://github.com/teheperinko541/Few-Shot-AIGI-Detector)的六个 `.pth`
权重只通过百度网盘发布，Hugging Face 上没有可核验副本；服务器的 GenImage train
Arrow 快照只有连续的 `985/1214` 个分片，不能据此训练并宣称论文复现。

## 数据完整性

| 数据视图 | 样本数 | 状态 |
| --- | ---: | --- |
| GenImage 六生成器公开测试视图 | 100,000 | 完成全量 FTNet/FTNet-T 评测 |
| GenImage OmniDFA 官方 zero-shot 聚合视图 | 56,000 | 6,000 real + 50,000 fake |
| OpenSDI 五生成器测试集 | 100,000 | 每个生成器 10,000 real + 10,000 fake |

OpenSDI 固定 Hugging Face revision 为
`7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`。实验结束后重新核验 52 个 LFS 文件，
共 18,266,080,332 bytes，全部 SHA-256 匹配，无失败项。

所有新增文件只写入 `/home/yabin/projects/fsaid`。`/data/DF-arrow-data/**` 仅被读取，
没有解包、补写、改名或删除服务器上的打包数据集。

## 论文协议结果

### GenImage FTNet Table 2

| 方法 | 论文 mAcc | 本次 mAcc | 差值 |
| --- | ---: | ---: | ---: |
| FTNet | 90.70 | 87.39 | -3.31 |
| FTNet-T | 94.20 | 89.46 | -4.74 |

逐生成器结果和差异分析见 [`REPRODUCTION_3090.md`](REPRODUCTION_3090.md)。
论文未发布实际 support 文件，因此这是固定 seed 的公开协议重跑，不是作者逐图片
划分的精确还原。

### OpenSDI FTNet Table 4

| 方法 | 论文 Acc | 本次 Acc | 论文 F1 | 本次 F1 |
| --- | ---: | ---: | ---: | ---: |
| FTNet | 79.94 | 83.08 | 77.83 | 84.33 |
| FTNet-T | 83.16 | 83.30 | 82.68 | 83.88 |

FTNet-T 的平均 Acc 与论文相差 `+0.14` 个百分点。逐生成器结果见
[`REPRODUCTION_OPENSDI_3090.md`](REPRODUCTION_OPENSDI_3090.md)。

### SynthBuster CLIPDet Table 4

| 生成器 | 论文AUC | 本次AUC | 差值 |
| --- | ---: | ---: | ---: |
| DALL-E 2 | 86.30 | 86.30 | 0.00 |
| DALL-E 3 | 92.90 | 92.87 | -0.03 |
| Midjourney v5 | 81.70 | 81.67 | -0.03 |
| Adobe Firefly | 87.20 | 87.49 | +0.29 |
| **宏平均** | **87.03** | **87.08** | **+0.06** |

官方5,000文件路径集合逐项一致，31/31个Arrow分片完整，物化图片全部通过记录内MD5。
详细协议和阈值指标见
[`REPRODUCTION_SYNTHBUSTER_3090.md`](REPRODUCTION_SYNTHBUSTER_3090.md)。

### GenImage 固定官方权重

| 方法 | Accuracy | Balanced Acc | Real Acc | Fake Acc | AP | ROC-AUC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIPDet | 70.180 | 81.937 | 96.900 | 66.974 | 99.263 | 94.906 | 80.043 |
| OmniDFA zero-shot | **95.857** | 96.111 | 96.433 | 95.788 | 99.755 | 98.655 | 97.635 |

OmniDFA 的 `Accuracy=95.857%` 精确复现论文报告的 `95.86%`。这里的 Accuracy 是
56,000 张图片上的样本加权准确率；Balanced Acc 是 real/fake accuracy 的均值。

## 额外统一评测

### OpenSDI 固定权重诊断

下表为五个生成器的宏平均：

| 方法 | Accuracy | AP | ROC-AUC | F1 |
| --- | ---: | ---: | ---: | ---: |
| CLIPDet | 63.511 | 86.444 | 86.075 | 43.872 |
| OmniDFA zero-shot | 76.515 | 81.097 | 79.805 | 72.002 |

这组结果用于同一框架下的额外诊断，不是原论文表格。OmniDFA zero-shot 的训练类
列表包含 `FLUX_Dev`，所以不能把它描述成 OpenSDI 五类全部严格 unseen。

### GenImage 六生成器留一诊断

每个生成器使用 10 real + 10 fake support、15 real + 15 fake query，seed=42。

| 方法 | Midjourney | GLIDE | ADM | SD | VQDM | BigGAN | 平均 Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FTNet | 96.67 | 96.67 | 90.00 | 76.67 | 96.67 | 100.00 | 92.78 |
| FTNet-T | 96.67 | 96.67 | 90.00 | 80.00 | 96.67 | 100.00 | 93.33 |

每个生成器的 query 总共只有 30 张，这一项用于检查留一流程，不能替代 FSD 论文的
完整测试。

### OpenSDI 持续学习主表

顺序为 `SD1.5 -> SD2.1 -> SDXL -> SD3 -> FLUX.1`。下表是最终阶段对全部五个
已见生成器的均值，再对 seeds 0/1/2 报告 `mean +/- std`：

| 方法 | K | Accuracy | AP | ROC-AUC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FTNet | 1 | 71.179 +/- 8.458 | 81.911 +/- 4.206 | 82.178 +/- 5.222 | 72.383 +/- 8.450 |
| FTNet | 5 | 79.596 +/- 2.505 | 88.762 +/- 1.230 | 89.981 +/- 0.751 | 82.105 +/- 1.355 |
| FTNet | 10 | 81.120 +/- 2.167 | 90.626 +/- 0.904 | 91.920 +/- 0.681 | 83.647 +/- 1.368 |
| FTNet-T | 1 | 73.112 +/- 5.866 | 81.685 +/- 3.789 | 81.437 +/- 5.207 | 73.035 +/- 7.905 |
| FTNet-T | 5 | 82.545 +/- 1.447 | 89.691 +/- 0.379 | 90.485 +/- 1.106 | 83.588 +/- 1.711 |
| FTNet-T | 10 | **84.816 +/- 0.719** | **92.222 +/- 0.099** | **93.185 +/- 0.186** | **86.001 +/- 0.579** |

完整结果含 2 个方法、3 个 K、3 个 seed 和 5 个增量阶段，共 270 条 lower-triangular
记录。FTNet-T 在 K=5 和 K=10 的最终 Accuracy 分别比 FTNet 高 2.949 和 3.696 个
百分点。

为避免对相同 10 万张 query 重复提取 CLIP 特征，这项使用冻结特征缓存运行。
smoke 对照中，标准 runner 与缓存 runner 的 30/30 条 classification accuracy 和
balanced accuracy 完全一致；极小样本、近似并列分数下 AP/AUC 的排序可能有浮点差异。

## 权重与版本

| 文件 | 来源版本 | SHA-256 |
| --- | --- | --- |
| OpenAI CLIP ViT-L/14 | OpenAI CLIP `d05afc4` | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| CLIPDet linear head | [官方 Git LFS HEAD](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) | `1c9c9f48a47d975bde67e9924983aa4e4ccda96a15e7e983b6305f0c56d9ee18` |
| CommonPool OpenCLIP backbone | HF revision `d979a50bf428edbb14b32c4dce7da2d49f7f46ed` | `d7d0ab41ad7025f048eab0b27bc5b9a2f4fa32601be9835bde8ae5daec9ea0d6` |
| OmniDFA zero-shot epoch 40 | [官方 HF](https://huggingface.co/MoeNew/OmniDFA) revision `b4c0f420461f154c0d71f1abc13fa01e11e4975b` | `616d3a0d3cc0b89d366b7f18fa531dd022f09fa2c15acd9aa4c8d7fee851b457` |

Hugging Face 文件均由 3090 直接通过 HF mirror 下载，没有经过本地电脑中转。

## 验证与产物

- `pytest`: 50 passed；
- `ruff check .`: passed；
- `python run.py verify`: FSD、FTNet、FTNet-T、CLIPDet、OmniDFA 五个来源锁全部通过；
- OpenSDI snapshot: 52/52 LFS SHA-256 passed。

小型原始结果表已归档到 [`reproductions/all_3090`](reproductions/all_3090)，服务器上的
完整 episode、adapter 和特征缓存保留在 `/home/yabin/projects/fsaid/outputs`。

## FSD 尚缺输入

要补齐 FSD，需要下面两种输入之一：

1. 官方六个 leave-one-generator-out `.pth` 权重；或
2. 完整 GenImage train 数据，至少补齐当前缺少的 229 个 Arrow 分片，并保证有足够
   项目磁盘空间完成物化与六模型训练。

在这两个条件都不满足时运行随机初始化 FSD 或用残缺训练集训练，会得到数值但没有
论文复现意义，因此本轮没有制造这类无效结果。
