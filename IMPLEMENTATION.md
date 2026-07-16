# Integrated implementation audit

本文档记录few-shot方法和固定权重评测方法如何被融合，以及哪些细节必须保持不变。

## 统一边界

以下部分由三个方法共享：

- manifest 和 image dataset；
- deterministic support/query sampling；
- image batch loading；
- method lifecycle：`adapt()` / `predict_fake_probability()`；
- continual stage runner；
- ACC、AP、AUC、F1 与结果序列化。

算法差异集中在根目录的 `methods/`，公共训练和评测入口为 `run.py`。

## FSD

参考实现 commit：`b545c05f3c927ef67c1b00f9a8badf3b68c5f4b3`。

- `timm` ImageNet-pretrained ResNet-50；
- classifier output 改为 1024维 metric embedding；
- 3-way episodic source training；
- 每类5 support、5 query；
- prototype 为 support embedding 均值；
- score 为 negative squared Euclidean distance；
- query loss 为 cross entropy；
- Adam，lr `1e-4`；
- StepLR，gamma `0.5`，step size `80,000`；
- 测试阶段不更新 encoder。

GenImage既可以使用官方ImageFolder逻辑视图，也可以直接使用完整Arrow快照。Arrow
索引只保存`shard,row`定位，不复制图片；ADM、BigGAN、GLIDE、Midjourney、VQDM
选取各自`ai`行，SD合并SD1.4、SD1.5与Wukong的`ai`行，real只选取SD1.4和
SD1.5的`nature`行。这改变存储访问方式，不改变episode类别或图像变换。

训练checkpoint额外保存优化器、scheduler、scaler和CPU/CUDA随机状态并支持续跑。
checkpoint先写入`.partial`文件，再通过原子替换发布，campaign只会选择完整命名的
checkpoint，并校验最终step、配置与SHA-256后创建评测别名。
恢复后DataLoader会重新建立shuffle迭代器，因此恢复运行不是中断前样本游标的逐位
延续。可选梯度累积用于单卡显存适配；启用时BatchNorm统计按micro-batch计算，结果
必须与官方默认batch运行分开标注。

已知 paper/code 差异：论文描述测试时 resize 后 center crop，但公开代码的验证
transform 只有 `CenterCrop(224) + ToTensor()`。框架采用公开代码行为。

## FTNet

算法与训练循环参考官方历史commit：
`139348d3a7627160cdfb1e4f537986bdf3c007f4`。

- OpenAI CLIP ViT-L/14；
- 输入分辨率224；
- 依次执行 visual patch embedding、class token、position embedding、
  `ln_pre` 和前12个 residual blocks；
- 取第12层 `x[0]` CLS token，不执行 `ln_post` 和 projection；
- feature 做 L2 normalization；
- keys 为 support features 的转置；
- values 为 binary one-hot labels；
- `affinity = query @ keys`；
- `logits = exp(-15 * (1 - affinity)) @ values`。

框架使用固定 commit 的 OpenAI CLIP 包，并在 `models.py` 中原生实现第12层
提前返回，因此不需要保留 FTNet 仓库修改过的 CLIP 源码副本。

## FTNet-T

后续官方HEAD删除了完整训练循环，因此这里以该公开历史实现和论文设置为准。
该历史代码读取 `finetune.*`，同commit配置却提供 `training.*`，所以不能把官方
仓库原样运行视为完整复现入口。

- 建立 `Linear(D, N, bias=False)` adapter；
- adapter weights 初始化为 cache keys 的转置；
- 只优化 adapter weights；
- CLIP 与 one-hot values 冻结；
- AdamW，lr `0.001`，eps `1e-4`，默认 weight decay；
- 每个 batch 更新 CosineAnnealingLR；
- cross-entropy；
- 20 epochs；
- seed 固定为40。

## Unified continual protocol

原论文并未使用完全相同的 continual protocol。统一框架只改变外围 episode
组织方式，不改变 feature、cache、prototype、loss 或 optimizer：

- 所有方法共享相同 support paths；
- support 不进入 query；
- FTNet cache 随阶段累计；
- FSD 为不同 generator 保留独立 prototype；
- 每阶段测试所有已见 generator。

因此，GenImage配置也是固定共享support的统一对照，不是FSD官方 `shuffle=True`
DataLoader逐batch episode的原样复现；OpenSDI配置用于统一continual comparison。
所有结果都会写出 `reproduction_scope`，避免把算法重跑误写成官方逐数复现。

## CLIPDet（evaluation-only）

参考实现commit：`c76ef7f5e158c5aba9e55b8b94ab0079720d281e`。

- OpenCLIP ViT-L/14 CommonPool checkpoint；
- 删除visual projection，使用next-to-last 1024维feature；
- CLIP normalization与224 bicubic resize/center crop；
- 加载作者发布的单层`ChannelLinear`权重；
- 输出原始LLR，官方判断规则为`LLR > 0`即fake；
- 不使用support，不提供训练入口。

## OmniDFA Detection（evaluation-only）

参考实现commit：`35b9052e83e05436682095818693493f79da9458`。

- 两个独立ConvNeXt-Small分别处理local crop和global resized crop；
- 验证时先执行官方的最小尺寸256处理，再分别random crop和resize/random crop；
- 拼接feature后通过`512 -> 128` MLP；
- feature与训练得到的real center均做L2 normalization；
- cosine similarity达到checkpoint内阈值时判为real，否则为fake；
- 框架仅将similarity取负，以统一“越大越fake”的分数方向，决策完全等价；
- 另行输出官方real-positive、20-threshold AP和balanced accuracy；
- few-shot attribution分支不属于真假检测，因此不接入本benchmark。

## 未接入的评测仓库

- LIDA当前只公开gallery/retrieval attribution评测，没有论文中的few-shot检测训练与检测头；
- Manifold官方代码只输出zero-shot criterion，没有发布few-shot fusion或二分类阈值；
- Fleet尚未发布代码。

这些方法不会通过猜阈值或补造训练过程接入二分类结果表。
