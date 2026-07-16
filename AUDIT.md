# End-to-end audit

审计日期：2026-07-16。审计对象为本仓库 `main` 分支，以及下列官方公开版本：

| 方法 | 官方来源 | 固定版本 |
|---|---|---|
| FSD | `teheperinko541/Few-Shot-AIGI-Detector` | `b545c05f3c927ef67c1b00f9a8badf3b68c5f4b3` |
| FTNet / FTNet-T | `zuiluorenjian/FTNet` | `139348d3a7627160cdfb1e4f537986bdf3c007f4` |
| CLIPDet | `grip-unina/ClipBased-SyntheticImageDetection` | `c76ef7f5e158c5aba9e55b8b94ab0079720d281e` |
| OmniDFA | `teheperinko541/OmniDFA` | `35b9052e83e05436682095818693493f79da9458` |

## 结论

本仓库可以作为统一重跑框架，但必须区分三件事：算法实现、公开实验协议、作者
实际使用的逐图片划分。只有三者都公开，才可以声称逐数复现官方表格。

RTX 3090上的GenImage/OpenSDI可执行核心实验已经全部完成，统一结果、权重哈希、
数据校验和未完成边界见[`REPRODUCTION_ALL_3090.md`](REPRODUCTION_ALL_3090.md)。
GenImage FTNet/FTNet-T Table 2的详细差异分析见
[`REPRODUCTION_3090.md`](REPRODUCTION_3090.md)。

| 方法 | 算法/推理实现 | 官方逐图片划分 | 可以声称什么 |
|---|---|---|---|
| FSD | 核心网络、loss、训练参数和prototype推理已对齐 | 未发布固定support文件；官方测试按batch重新组成episode | 官方算法在固定共享support协议上的重跑 |
| FTNet | 第12层CLIP特征、cache公式、温度15已对齐 | Table 1–4 support文件未发布 | 公开论文协议复现，不能声称逐图片复原 |
| FTNet-T | 论文参数与官方历史训练循环已对齐 | 未发布；当前HEAD还删除了完整训练循环 | 论文设置+历史公开循环的重跑，不能保证官方表格逐数一致 |
| CLIPDet | 使用官方固定backbone、预处理、线性头和阈值 | 不依赖few-shot划分 | 给定同一权重和图片时的官方推理复现 |
| OmniDFA Detection | 架构、验证变换、checkpoint threshold已对齐 | 官方提供GenImage/Chameleon类列表 | 官方固定权重推理；OpenSDI仅为额外诊断 |

## 官方仓库自身的限制

### FSD

- 官方 `test.py` 的DataLoader使用 `shuffle=True`；每个batch包含support和query，
  因而support会随batch变化。
- 本框架为了所有few-shot方法共享相同support，使用一个固定support集合评测剩余
  query。这是有意的统一协议，不是FSD官方表格的原始episode组织。
- 官方代码默认5 support/15 query，但论文比较表也报告10-shot；CLI参数允许修改
  support数量。

### FTNet / FTNet-T

- 论文明确披露ViT-L/14、224输入、第12层、温度15、4/8-shot，以及FTNet-T的
  AdamW、lr 0.001和20 epochs。
- 官方历史commit包含完整cache与adapter训练循环，但其代码读取 `finetune.*`，
  同commit的 `config.yaml` 却提供 `training.*`，不能原样启动。
- 官方当前HEAD的 `FTNet.py` 期望 `cache_images`、`cache_labels`、`test_datasets`
  等字段，当前 `config.yaml` 仍是另一套schema；`FTNet-T.py` 的 `run()` 只打印
  initialized，不再执行完整训练。
- 因此本框架保留公开公式和历史训练循环，但绝不把未发布的support文件或缺失
  配置猜成“作者原始划分”。

### CLIPDet

- 官方只发布推理代码和固定权重，不发布论文中的训练入口。
- 本框架的 `clipdet` 仅为 `evaluation_only`，不会假装成few-shot训练方法。

### OmniDFA

- few-shot部分是生成器来源归因；真假检测分支是固定权重authenticity detection。
- 官方authenticity指标把real当正类，并以real/fake accuracy的均值作为Acc。
  本框架同时输出统一fake-positive指标，以及字段
  `official_balanced_accuracy`、`official_real_positive_average_precision_20`。
- 官方发布了三个fold权重和 `OmniDFA_zero_shot_epoch[40].pth`。GenImage应使用
  zero-shot权重。该权重训练列表包含FLUX_Dev，所以OpenSDI上的OmniDFA结果不应
  被写成“五个生成器全部unseen”。

## 数据与协议审计

- manifest全局拒绝重复绝对路径。
- 每个stage、每个label都必须同时有足够support和非空query。
- 显式support/query不能与pool混用。
- support/query按路径标识和seed进行稳定SHA-256排序，与CSV行序无关。
- source generators与target stages存在名称交集时直接报错。
- `shots: 0`只允许固定权重evaluation-only方法，用于保证所有图片都进入query；
  FSD、FTNet或FTNet-T使用0-shot会报错。
- 每次运行把实际support/query路径写入 `episode.json`，作为可核验实验凭据。

如果要求不同K使用完全相同的query，应在manifest中显式标记support/query。pool
模式会从同一有序池中取前K张support，因此K变化时query也会相应变化。

## 本轮修复

1. 补回OmniDFA验证代码中对短边小于256图片的官方最小尺寸处理；此前小图可能
   crop失败，224–255像素图片的输入也与官方不一致。
2. 固定权重评测从伪K-shot改为真正0-shot，不再无意义地丢掉K张query图片。
3. OmniDFA改用官方zero-shot checkpoint配置，并标明OpenSDI的FLUX重叠风险。
4. 增加方法级 `reproduction_scope`，每条结果都记录其可复现边界。
5. 增加OmniDFA官方real-positive指标，同时保留统一fake-positive指标。
6. 修复不同方法拥有额外指标列时CSV写出失败的问题。
7. 增加空query、非法label、NaN/Inf score、重复方法、0-shot误用和配置schema检查。
8. 增加GenImage固定权重评测配置和官方zero-shot权重路径。
9. 拒绝重复shot/seed、字符串伪布尔值、未知方法参数和损坏的manifest行，避免静默
   重复实验或错误配置进入长任务。
10. FSD与OmniDFA加载完整官方checkpoint时不再预先下载一份会被立刻覆盖的timm
    预训练权重；FSD source training仍按官方设置从ImageNet预训练权重开始。
11. 增加仓库级`.gitignore`，隔离数据、checkpoint、输出、Python缓存和本地环境。
12. 增加官方GenImage测试包的BigGAN/GLIDE抽取、六生成器合并和清单生成工具。
13. 增加显式4-shot support侧车清单；本次48张support已与实际`episode.json`
    逐项核对一致。
14. 在RTX 3090上完成10万张GenImage的FTNet与FTNet-T全量重跑，并保存逐生成器
    指标、汇总表和论文差值。
15. 直接通过HF mirror下载并锁定OmniDFA zero-shot权重与CommonPool OpenCLIP
    backbone；CLIPDet head使用官方Git LFS文件，三个文件均记录SHA-256。
16. 完成GenImage 56,000张官方zero-shot聚合评测；OmniDFA得到95.8571% Accuracy，
    四舍五入与论文Table 4的95.86%一致。
17. 完成OpenSDI固定权重诊断，以及FTNet/FTNet-T的K=1/5/10、三个seed、五阶段
    continual矩阵，共270条结果。
18. 增加冻结CLIP特征缓存runner；标准与缓存smoke的30/30条classification accuracy
    和balanced accuracy一致。
19. 新增统一Balanced Accuracy字段，并保持对旧式结果记录的兼容。
20. 找到并只读接入服务器现有SynthBuster 31分片Arrow快照；官方5,000文件路径集合
    哈希完全一致，所有物化图片通过记录内MD5。
21. 完成CLIPDet四个商业生成器评测；AUC宏平均87.08%，论文四项宏平均87.03%。

## 仍然无法由代码消除的限制

- FTNet论文表格和FSD论文表格的逐图片support列表没有公开。
- Git仓库不内置数据集和大权重；本次服务器项目目录已准备CLIP、CLIPDet、OmniDFA
  权重并完成实测，但换机器仍需按README重新下载。
- FTNet/FTNet-T、CLIPDet和OmniDFA已在真实GenImage/OpenSDI与GPU上完成验证。
  FSD仍缺官方六个`.pth`权重；服务器GenImage train Arrow快照也只有985/1214个
  分片，不能用于有效论文训练。仓库测试覆盖公式、layer截断、划分、防泄漏、
  runner、指标、官方验证resize和全部模块导入。
- FTNet官方GitHub仓库没有根LICENSE文件；公开使用或再分发前应向作者确认授权。

以上限制会保留在文档中，不以近似实现或猜测划分掩盖。
