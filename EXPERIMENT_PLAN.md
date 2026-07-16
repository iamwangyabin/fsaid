# 二分类实验复现计划

本文档定义本项目后续需要完成的实验。研究主任务统一为 AI 生成图像的
`real/fake` 二分类检测；OmniDFA 的 5-way/15-way 生成器来源归因不纳入本计划。

## 1. 状态说明

- `[x]`：框架代码或配置已具备，并已通过单元测试。
- `[ ]`：仍需数据、权重、GPU实验或额外实现，不能视为已经得到论文数值。

当前代码已融合 FSD、FTNet、FTNet-T、CLIPDet evaluation-only 和 OmniDFA
Detection evaluation-only。Git仓库不跟踪数据集和大模型权重；3090项目目录已准备
GenImage公开测试视图和完整OpenSDI测试集，并已完成当前输入允许的全量实验，但
尚未接入RACE。结果见[`REPRODUCTION_ALL_3090.md`](REPRODUCTION_ALL_3090.md)。

实验必须分成两组报告：

1. **原论文协议验证**：按每篇论文公开的协议检查实现是否正确；
2. **统一公平比较**：所有 few-shot 方法使用相同 support/query、K 和随机种子。

两组结果不得混在同一列中，也不得把统一协议结果标成论文表格的逐数复现。

## 2. 方法范围

| 方法 | 二分类角色 | 新阶段是否使用support | 当前状态 |
|---|---|---:|---|
| FSD | few-shot prototype detector | 是 | `[x]` 已融合；`[ ]` 待训练/放置权重 |
| FTNet | training-free feature cache | 是 | `[x]` 已融合；`[x]` GenImage/OpenSDI实测 |
| FTNet-T | learnable feature cache | 是 | `[x]` 已融合；`[x]` GenImage/OpenSDI实测 |
| CLIPDet | 固定权重检测参考 | 否，使用K=0 | `[x]` 已融合；`[x]` GenImage/OpenSDI实测 |
| OmniDFA Detection | 固定权重检测参考 | 否，使用K=0 | `[x]` 已融合；`[x]` GenImage/OpenSDI实测 |
| RACE | 待比较的目标方法 | 是 | `[ ]` 尚未接入本仓库 |

CLIPDet 和 OmniDFA Detection 不得描述成 few-shot adaptation 方法。FSD 虽然在源
训练时区分多个生成器类别，但正式评测仍为 target fake 与 real 的二分类。

## 3. 数据准备

### 3.1 GenImage（核心、必须）

准备 ADM、GLIDE、Midjourney、SD1.4、SD1.5、Wukong、VQDM、BigGAN 和对应的
ImageNet real images，并保留官方 train/val 划分。

需要生成两套逻辑视图：

1. **FSD论文视图**：把 SD1.4、SD1.5、Wukong 合并为 `SD`，形成 Midjourney、
   GLIDE、ADM、SD、VQDM、BigGAN 六个fake generator classes；
2. **OpenSDI无泄漏source视图**：如果后续target包含SD1.5，FSD源训练必须排除
   整个`SD`合并类，只使用ADM、GLIDE、Midjourney、VQDM、BigGAN和real。

任务：

- [ ] 下载完整GenImage；
- [ ] 核验官方train/val目录和各子集图片数量；
- [ ] 检测损坏图片和重复绝对路径；
- [ ] 生成`data/manifests/genimage.csv`；
- [ ] 生成FSD论文协议和无泄漏source的统计报告。

当前已完成10万张官方测试视图及5.6万张OmniDFA zero-shot聚合视图。服务器上的
train Arrow快照只有985/1214个连续分片，因此上述“完整GenImage”训练数据任务仍
保持未完成，不能用现有快照训练FSD论文模型。

### 3.2 OpenSDI（核心、必须）

统一增量顺序固定为：

```text
SD1.5 -> SD2.1 -> SDXL -> SD3 -> FLUX.1
```

每个stage都必须包含real、fake以及互不重叠的support/query候选图片。

任务：

- [x] 下载五个生成器的完整官方测试集；
- [x] 核验每个生成器real/fake数量；
- [x] 全量图片通过CLIP解码，manifest绝对路径无重复；
- [ ] 额外执行跨generator的内容级重复分析；
- [x] 生成`data/manifests/opensdi_table4_seed40.csv`；
- [x] 固化Table 4的K=8、seed=40 support/query清单；
- [ ] shot消融前固化其余K和seed的清单。

### 3.3 扩展数据（完成核心实验后）

| 数据集 | 目的 | 优先级 |
|---|---|---|
| UniversalFakeDetect | FTNet/FTNet-T Table 3及shot消融 | 高，但下载源可能不稳定 |
| Chameleon | OmniDFA严格zero-shot二分类 | 中 |
| SynthBuster | CLIPDet商业生成器和鲁棒性 | `[x]` 原图官方评测；鲁棒性待跑 |
| OmniFake | OmniDFA三折真假检测 | 低，数据和权重开销大 |

## 4. 数据完整性与防泄漏

在开始GPU实验前必须完成以下检查：

- [x] manifest重复路径检查；
- [x] support/query互斥检查；
- [x] real/fake label合法性检查；
- [x] 空query和K-shot数量检查；
- [x] source/target generator名称重叠检查；
- [x] 已执行实验数据视图的官方预期图片数量检查；
- [x] GenImage与OpenSDI全量评测完成图片解码；
- [x] OpenSDI记录固定revision、镜像来源、文件数量和52个LFS SHA-256；
- [ ] 为其余扩展数据补齐同等级checksum记录；
- [x] 已完成运行保存实际`episode.json`，support清单与指标已归档。

若要让不同K共享完全相同的query，必须使用显式`support/query`划分。`pool`模式会
随K增加而从query中移走更多图片，不适合做严格的跨K逐图片比较。

## 5. 权重准备

### 5.1 FSD

- [ ] 准备六个GenImage leave-one-generator-out权重：Midjourney、GLIDE、ADM、
  SD、VQDM、BigGAN各一个；
- [ ] 使用无泄漏GenImage source重新训练
  `checkpoints/fsd/source_without_opensdi.pth`；
- [ ] 保存训练配置、随机种子、最终step和checkpoint SHA-256。

无泄漏OpenSDI source权重与官方其他leave-one-out权重不是同一个实验，不能相互
替代。

### 5.2 FTNet / FTNet-T

- [x] 下载并固定OpenAI CLIP ViT-L/14；
- [x] 记录模型文件SHA-256和OpenAI CLIP代码版本。

FTNet没有额外方法checkpoint。FTNet-T的adapter在每个support episode中重新初始化
并训练，不预先下载。

### 5.3 CLIPDet

- [x] 下载官方`clipdet_latent10k_plus/weights.pth`；
- [x] 下载CommonPool OpenCLIP ViT-L/14 backbone；
- [x] 记录两个文件的来源和SHA-256。

### 5.4 OmniDFA Detection

- [x] 核心zero-shot评测下载`OmniDFA_zero_shot_epoch[40].pth`；
- [ ] 如果执行OmniFake三折，再下载三个fold checkpoints；
- [x] 记录权重适用的训练类列表，避免把有重叠的OpenSDI结果写成严格unseen。

## 6. 原论文协议验证

### 6.1 FSD

数据使用GenImage论文视图。

- [ ] 分别排除Midjourney、GLIDE、ADM、SD、VQDM、BigGAN，训练六个metric
  encoders；
- [ ] 复现六个unseen generator的10-shot ACC/AP；
- [ ] 复现六个unseen generator的zero-shot ACC/AP；
- [ ] 输出完整`6 training models x 6 test generators`的ACC/AP矩阵；
- [ ] 运行shot数量消融；
- [ ] 运行多生成器metric training与普通real/fake二分类训练的ADM对比；
- [ ] 生成六个模型的t-SNE及ADM对比可视化。

训练设置固定为ResNet-50、1024维、3-way、每类5 support/5 query、Adam `1e-4`、
StepLR `step_size=80000`/`gamma=0.5`、200,000 steps。

论文没有发布逐episode固定文件表；结果需要注明随机性，不能承诺逐数一致。

### 6.2 FTNet / FTNet-T

固定CLIP ViT-L/14、224x224、第12层CLS、temperature 15。FTNet-T固定AdamW、
learning rate `0.001`、20 epochs。

#### GenImage

- [x] 4-shot FTNet与FTNet-T，报告六个生成器ACC和mAcc；
- [ ] 运行single-source intra-domain/cross-domain协议；
- [x] 运行cross-generator validation协议。

#### UniversalFakeDetect

- [ ] 4-shot FTNet与FTNet-T；
- [ ] 报告19个测试子集ACC和mAcc。

#### OpenSDI

- [x] 8-shot FTNet与FTNet-T；
- [x] 分别报告SD1.5、SD2.1、SDXL、SD3、FLUX.1的ACC和F1；
- [x] 报告平均ACC和平均F1。

#### 消融与分析

- [ ] 三个数据集的shot数量消融；
- [ ] CLIP第6、12、18、24层消融；
- [ ] FTNet与FTNet-T对比；
- [ ] adapter epochs与temperature敏感性；
- [ ] GenImage各层t-SNE和FLUX.1 nearest-neighbor可视化。

FTNet论文没有发布各表实际使用的逐图片support清单。必须保存本项目抽到的清单，
并把结果描述为“公开论文协议重跑”。

### 6.3 CLIPDet固定权重验证

- [x] 使用官方head、CommonPool backbone、预处理和阈值；
- [x] 在GenImage和SynthBuster上报告ROC-AUC、AP、ACC、real/fake ACC、F1；
- [ ] 运行JPEG、WebP和resize鲁棒性；
- [ ] 将结果标记为`evaluation_only`和K=0。

本项目不把未公开完整训练入口的paired-data SVM训练称作严格复现。

### 6.4 OmniDFA Detection固定权重验证

- [x] 使用zero-shot checkpoint评测GenImage官方八fake类加一real类聚合视图；
- [ ] 报告per-generator ACC、F-Acc、R-Acc、balanced ACC、AP和F1；
- [ ] 使用同一zero-shot权重评测Chameleon；
- [ ] 可选执行OmniFake三个fold的unseen-generator真假检测；
- [ ] 将固定权重结果标记为`evaluation_only`和K=0。

## 7. 统一few-shot continual主实验

### 7.1 核心协议

```text
source: GenImage non-overlapping generators
targets: SD1.5 -> SD2.1 -> SDXL -> SD3 -> FLUX.1
K: 1, 5, 10
seeds: 0, 1, 2
evaluation_scope: all seen generators
```

每个stage执行：

1. 抽取K张当前generator fake和K张real作为support；
2. support不进入任何query；
3. 所有few-shot方法使用完全相同的support；
4. 更新当前方法状态；
5. 测试当前及此前所有generator；
6. 保存episode、模型产物、预测分数、指标和运行环境。

固定权重的CLIPDet和OmniDFA使用K=0，不能为了形式统一而从query删除它们不会使用
的support图片。它们应放在独立的zero-shot参考列，或在所有方法共同query上额外运行
一次并清楚标注。

### 7.2 方法更新规则

| 方法 | Stage 0 | 新stage操作 |
|---|---|---|
| FSD | 训练无泄漏source encoder | 计算新的real/fake prototype |
| FTNet | 无训练 | support加入累计cache |
| FTNet-T | 无训练 | support加入cache并训练adapter |
| RACE | 训练初始模型 | discrepancy判断和residual expert更新 |
| CLIPDet | 固定权重 | 不更新 |
| OmniDFA Detection | 固定权重 | 不更新 |

任务：

- [x] FSD/FTNet/FTNet-T continual runner；
- [x] CLIPDet/OmniDFA evaluation-only runner；
- [ ] 接入RACE并遵守同一manifest/episode接口；
- [x] FTNet/FTNet-T完成K=1/5/10、三个seed的完整GPU运行；
- [x] FTNet/FTNet-T所有已见generator的lower-triangular结果矩阵；
- [ ] FSD需官方权重或完整GenImage train后补跑同一矩阵。

## 8. 指标

### 8.1 所有二分类方法统一指标

- Accuracy；
- Balanced Accuracy；
- Real Accuracy；
- Fake Accuracy；
- Average Precision；
- ROC-AUC；
- F1；
- query样本数。

当前runner已输出Accuracy、Balanced Accuracy、Real/Fake Accuracy、AP、ROC-AUC、
F1和样本数。OmniDFA同时保留论文定义的real-positive AP。

### 8.2 Continual指标

- 每stage当前generator性能；
- 每stage所有已见generator平均性能；
- Final Average Accuracy；
- Average Seen Accuracy；
- Forgetting；
- Backward Transfer（BWT）；
- 每stage更新时间；
- 推理吞吐与峰值显存；
- 可训练参数量；
- support/cache大小。

任务：

- [x] lower-triangular逐stage基础结果；
- [ ] Forgetting、BWT及其单元测试；
- [ ] 时间、显存、参数量和cache统计；
- [ ] 所有随机实验报告`mean +/- std`。

## 9. 鲁棒性扩展

核心主表完成后，使用相同query语义分别测试：

- [ ] JPEG quality；
- [ ] WebP quality；
- [ ] resize；
- [ ] Gaussian blur；
- [ ] random crop；
- [ ] 不同输入分辨率；
- [ ] support退化/query原图；
- [ ] support原图/query退化。

重点观察FSD prototype稳定性、FTNet cache敏感性、FTNet-T对support的过拟合、RACE
持续适配能力，以及两个固定检测器的zero-shot鲁棒性。

## 10. 结果与交付物

最终至少生成：

1. FSD GenImage论文协议表；
2. FTNet/FTNet-T GenImage论文协议表；
3. FTNet/FTNet-T OpenSDI论文协议表；
4. UniversalFakeDetect扩展表；
5. 统一OpenSDI continual主表；
6. K=1/5/10消融表；
7. Forgetting/BWT表；
8. 参数量、时间、显存和cache开销表；
9. 鲁棒性表；
10. 原论文协议与统一协议差异说明。

每张表必须附带：配置文件、git commit、数据manifest、episode文件、checkpoint
SHA-256、随机种子和原始逐图片预测结果。

## 11. 推荐执行顺序

1. [ ] 下载并完整核验GenImage；
2. [ ] 下载并完整核验OpenSDI；
3. [ ] 生成和冻结manifests及episodes；
4. [ ] 下载CLIP、OpenCLIP、CLIPDet和OmniDFA权重；
5. [ ] 训练六个FSD论文模型；
6. [ ] 训练FSD OpenSDI无泄漏source模型；
7. [ ] 完成FSD论文协议验证；
8. [ ] 完成FTNet/FTNet-T GenImage和OpenSDI协议验证；
9. [ ] 完成CLIPDet与OmniDFA固定权重评测；
10. [ ] 接入并验证RACE；
11. [ ] 运行统一K=1/5/10、三个seed主实验；
12. [ ] 计算continual、效率和资源指标；
13. [ ] 运行扩展数据与鲁棒性实验；
14. [ ] 汇总表格并归档全部复现凭据。

## 12. 完成标准

核心二分类实验完成必须同时满足：

- GenImage与OpenSDI通过完整性检查；
- FSD、FTNet、FTNet-T和RACE完成所有K与seed；
- 所有few-shot方法使用逐图片完全一致的support/query；
- 固定检测器以K=0单独标注；
- 所有结果能够从保存的config、manifest、episode和checkpoint重新运行；
- 对未公开support列表的论文不声称逐数复原；
- 主表、消融、continual指标和资源开销均有原始记录。

核心范围为：

```text
GenImage + OpenSDI + FSD + FTNet + FTNet-T + RACE
```

UniversalFakeDetect、Chameleon、SynthBuster和OmniFake属于完成核心范围后的扩展。
