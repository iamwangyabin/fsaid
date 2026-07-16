# CLIPDet SynthBuster reproduction on RTX 3090

执行日期：2026-07-16。

## 范围

本次使用CLIPDet官方`clipdet_latent10k_plus`权重和CommonPool OpenCLIP ViT-L/14，
复现论文Table 4中四个商业生成器的原图AUC：DALL-E 2、DALL-E 3、Midjourney v5
和Adobe Firefly。

官方依据：

- [CLIPDet官方仓库](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)；
- [CVPRW 2024论文](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Cozzolino_Raising_the_Bar_of_AI-generated_Image_Detection_with_CLIP_CVPRW_2024_paper.pdf)。

## 数据与协议

- 只读Arrow快照：`/data/DF-arrow-data/synthbuster`，约15 GB；
- 31/31个Arrow分片均存在，共10,000张唯一图片；
- 官方`commercial_tools.csv`视图为1,000张RAISE real和四组各1,000张fake；
- 每个生成器分别评测同一批1,000张real和自己的1,000张fake；
- 官方5,000文件路径集合SHA-256：
  `45ab1a5252079542757d10d5490eb57234dd70d6fccc906a2fdf5d04924af565`；
- 5,000张物化图片全部通过Arrow记录中的逐图片MD5；
- LLR阈值为0，论文主要比较不依赖阈值的ROC-AUC。

3090直连GitHub下载官方CSV时超时，因此使用快照自带`test.json`生成等价索引。
生成前将其5,000文件名排序后与固定commit的官方CSV比较，路径集合SHA-256完全一致。

所有物化图片、hardlink视图、manifest和结果只写入
`/home/yabin/projects/fsaid`。打包Arrow目录没有被修改。

## 论文AUC对比

| 生成器 | 论文AUC | 本次AUC | 差值 |
| --- | ---: | ---: | ---: |
| DALL-E 2 | 86.30 | 86.30 | 0.00 |
| DALL-E 3 | 92.90 | 92.87 | -0.03 |
| Midjourney v5 | 81.70 | 81.67 | -0.03 |
| Adobe Firefly | 87.20 | 87.49 | +0.29 |
| **宏平均** | **87.03** | **87.08** | **+0.06** |

前三个生成器按论文的一位小数报告时完全一致；Firefly高0.29个百分点。四项宏平均
只差0.06个百分点，可以视为官方固定推理的成功复现。

## 额外指标

| 生成器 | Accuracy | Real Acc | Fake Acc | AP | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DALL-E 2 | 67.95 | 97.00 | 38.90 | 86.39 | 54.83 |
| DALL-E 3 | 74.00 | 97.00 | 51.00 | 92.09 | 66.23 |
| Midjourney v5 | 62.70 | 97.00 | 28.40 | 81.18 | 43.23 |
| Adobe Firefly | 64.60 | 97.00 | 32.20 | 86.28 | 47.63 |
| **宏平均** | **67.31** | **97.00** | **37.63** | **86.49** | **52.98** |

固定阈值对real明显偏保守，因此Accuracy和F1低于AUC表现。这不是重新校准后的结果，
而是保留官方LLR阈值0的真实表现。

## 运行与验证

- GPU推理时间：5分30秒；
- 峰值resident memory：约4.17 GB；
- `pytest`: 50 passed；
- `ruff check .`: passed；
- `python run.py verify`: 五个方法来源锁全部通过。

服务器结果位于：

`/home/yabin/projects/fsaid/outputs/synthbuster_clipdet_official/`

小型结果表和清单摘要归档在：

`reproductions/all_3090/synthbuster_clipdet_official/`
