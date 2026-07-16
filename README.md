# Few-shot AIGC Detection Benchmark

一个统一的 few-shot AI-generated image detection 实验框架，原生包含：

- **FSD**
- **FTNet**
- **FTNet-T**
- **CLIPDet**（evaluation-only）
- **OmniDFA Detection**（evaluation-only）

3090服务器上的全量可执行实验、权重哈希、数据校验和FSD缺口见
[`REPRODUCTION_ALL_3090.md`](REPRODUCTION_ALL_3090.md)。两个论文协议的详细结果见
[`REPRODUCTION_3090.md`](REPRODUCTION_3090.md)（GenImage）和
[`REPRODUCTION_OPENSDI_3090.md`](REPRODUCTION_OPENSDI_3090.md)（OpenSDI）；
CLIPDet商业生成器复现见
[`REPRODUCTION_SYNTHBUSTER_3090.md`](REPRODUCTION_SYNTHBUSTER_3090.md)。

所有方法共享同一套数据读取、support/query 划分、backbone 接口、continual
protocol、指标、日志与结果格式。项目中不存在三个外挂官方仓库，也不存在
`external/fsd`、`external/ftnet` 或通过 subprocess 调官方脚本的结构。

## 代码结构

```text
fewshot-aigc-benchmark/
├── configs/              # 实验配置
├── methods/              # few-shot方法与固定权重评测方法
├── reproductions/        # 已完成的真实数据复现实验结果
├── tests/                # 正确性与实现锁测试
├── run.py                # 唯一命令入口及统一实验 runner
├── AUDIT.md              # 官方源码、协议和可复现边界审计
├── EXPERIMENT_PLAN.md     # 二分类论文复现与统一实验执行计划
├── NOTICE.md             # 第三方方法与权重许可边界
├── data.py               # manifest、episode 与防泄漏检查
├── models.py             # ResNet-50 与 CLIP ViT-L/14
├── config.py             # YAML 配置读取
├── utils.py              # 指标、异常与实现来源锁
└── train_fsd.py          # FSD source episodic training
```

## 方法实现

| 方法 | Backbone | Support 更新 | 推理 |
|---|---|---|---|
| FSD | ImageNet ResNet-50，1024维 | 每个 generator 计算 real/fake prototype | squared Euclidean nearest prototype |
| FTNet | OpenAI CLIP ViT-L/14 第12层 CLS | support feature 写入 key-value cache | exponential affinity × one-hot labels |
| FTNet-T | 与 FTNet 相同 | cache key 初始化线性 adapter，训练20 epochs | learned-key cache inference |
| CLIPDet | CommonPool OpenCLIP ViT-L/14 | 不更新（固定官方线性头） | 官方LLR，阈值0 |
| OmniDFA Detection | Twin ConvNeXt-Small | 不更新（固定官方checkpoint） | 与real center的余弦阈值 |

FTNet-T 只更新 cache keys；CLIP 和 label values 始终冻结。优化器为 AdamW，
learning rate `0.001`，`eps=1e-4`，CosineAnnealingLR，20 epochs。

具体公式和源码对应关系见 [实现审计](IMPLEMENTATION.md)。
从官方仓库、数据协议到测试结果的完整核查见 [端到端审计](AUDIT.md)。
后续需要准备的数据、权重、论文协议、统一主表、指标和执行顺序见
[二分类实验复现计划](EXPERIMENT_PLAN.md)。计划中的复选框明确区分“代码已具备”与
“已经在真实数据上得到数值”，未勾选项目不能视为已完成实验。

CLIPDet和OmniDFA在本框架中明确标记为 `evaluation_only`。它们可以评测统一
query，但不会使用K-shot support更新参数，也不能冒充few-shot adaptation方法。

## 安装

```bash
conda env create -f environment.yml
conda activate fewshot-aigcd

python run.py verify
pytest
```

环境锁定 PyTorch 2.5.1、CUDA 12.1，并从 OpenAI CLIP 官方 commit
`d05afc4` 安装 CLIP。CLIP 权重首次运行时下载，之后使用本地 cache。

不要混用不同来源的 `torch` 和 `torchvision` wheel；二者必须来自同一PyTorch/
CUDA channel，否则可能出现 `torchvision::nms` 不存在。

## 数据格式

统一输入为 CSV：

```csv
path,label,generator,split
../OpenSDI/SD1.5/real/a.png,0,SD1.5,pool
../OpenSDI/SD1.5/fake/b.png,1,SD1.5,pool
```

- `label=0`：real；`label=1`：fake。
- `generator`：所属增量阶段。
- `split`：`pool`，或者显式的 `support` / `query`。
- 推荐使用相对路径，保证换机器后 support sampling 不变。

若要比较不同K且保持query逐图片完全一致，请使用显式support/query。pool模式会
取稳定排序后的前K张作为support，所以K变化时query集合也会变化。

也可以从文件夹生成 manifest：

```text
data/continual/
├── SD1.5/{real,fake}/
├── SD2.1/{real,fake}/
├── SDXL/{real,fake}/
├── SD3/{real,fake}/
└── FLUX.1/{real,fake}/
```

```bash
python run.py index \
  --data-root data/continual \
  --output data/manifests/opensdi.csv
```

## 运行统一实验

```bash
python run.py validate --config configs/opensdi_continual.yaml
python run.py run --config configs/opensdi_continual.yaml --method all --dry-run
python run.py run --config configs/opensdi_continual.yaml --method all
```

FTNet论文式OpenSDI联合cache协议：

```bash
python run.py run --config configs/opensdi_ftnet_paper.yaml --method all
```

该配置对五个生成器分别抽K real/K fake，合并为一个cache，再统一评测各生成器。
论文没有发布Table 4的具体support文件列表，因此框架固定seed并把实际文件名保存
到`episode.json`。这是公开协议复现，不能宣称作者原始split的逐图片复原。

Hugging Face Parquet快照采用只读准备流程，原快照、图片工作目录和manifest目录
必须分开。脚本会拒绝向快照内部写入，并可按镜像元数据核验LFS SHA-256：

```bash
python scripts/verify_hf_snapshot.py \
  --snapshot-root data/huggingface/nebula/OpenSDI_test \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df

python scripts/prepare_opensdi.py \
  --snapshot-root data/huggingface/nebula/OpenSDI_test \
  --work-root data/work/opensdi_test_images \
  --output-manifest data/manifests/opensdi_table4_seed40.csv \
  --explicit-shots 8 --seed 40 \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df
```

持续学习和固定权重评测需要保留全量候选池，不预先移除support：

```bash
python scripts/prepare_opensdi.py \
  --snapshot-root data/huggingface/nebula/OpenSDI_test \
  --work-root data/work/opensdi_test_images \
  --output-manifest data/manifests/opensdi.csv \
  --pool \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df
```

单独运行：

```bash
python run.py run --config configs/opensdi_continual.yaml --method fsd
python run.py run --config configs/opensdi_continual.yaml --method ftnet
python run.py run --config configs/opensdi_continual.yaml --method ftnet_t
```

## 固定权重评测方法

CLIPDet的官方仓库通过Git LFS发布`clipdet_latent10k_plus/weights.pth`；将其放到：

```text
checkpoints/clipdet/clipdet_latent10k_plus.pth
```

其CommonPool OpenCLIP主干放到：

```text
checkpoints/backbones/open_clip_pytorch_model.bin
```

OmniDFA官方发布三个fold checkpoints和一个zero-shot checkpoint。GenImage评测
使用zero-shot权重并放到：

```text
checkpoints/omnidfa/OmniDFA_zero_shot_epoch40.pth
```

先按OmniDFA发布的八个fake类和一个real类生成官方聚合清单，再运行GenImage
固定权重评测：

```bash
python scripts/prepare_genimage.py \
  --dataset-root /path/to/GenImage \
  --output-manifest data/manifests/genimage_official_zero_shot.csv \
  --view omnidfa-zero-shot --filesystem-order
```

```bash
python run.py run --config configs/genimage_evaluation_only.yaml --method all
```

该配置对应OmniDFA官方aggregate authenticity evaluation：把官方zero-shot real/fake
列表转换为一个manifest，并将所有行的 `generator` 统一写成 `GenImage`。如果需要
逐生成器结果，可另建manifest和stages，但那属于统一框架诊断而非官方聚合协议。

在OpenSDI上运行额外诊断：

```bash
python run.py run --config configs/opensdi_evaluation_only.yaml --method all
```

从只读Hugging Face Arrow快照准备并运行CLIPDet官方SynthBuster商业生成器评测：

```bash
python scripts/prepare_synthbuster.py \
  --snapshot-root /data/DF-arrow-data/synthbuster \
  --work-root data/work/synthbuster_clipdet_official \
  --official-csv data/reference/clipdet_commercial_tools.csv \
  --output-manifest data/manifests/synthbuster_clipdet_official.csv

python run.py run --config configs/synthbuster_clipdet_official.yaml --method clipdet
```

准备脚本要求官方5,000文件路径集合SHA-256匹配固定值，并核对每张图片的内置MD5；
快照目录、物化工作目录和manifest目录必须分开。

固定权重配置使用 `shots: [0]`，因此不会为了一个并未使用的support集合而从query
中删除图片。OmniDFA zero-shot训练列表包含FLUX_Dev，所以它的OpenSDI结果不能
写成“五个生成器全部unseen”；严格官方zero-shot目标是GenImage和Chameleon。

权重来源：

- [CLIPDet官方仓库](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)
- [OmniDFA官方仓库](https://github.com/teheperinko541/OmniDFA)

在 stage `t`：

- FTNet / FTNet-T 使用 `0..t` 的累计 support cache；
- FSD 保存每个已见 generator 的 binary prototype；
- 测试所有已见 generator，得到 lower-triangular continual matrix；
- 三个方法使用完全相同的 K real/K fake support。

多shot、多seed的全量continual评测会重复读取相同query。可使用冻结CLIP特征缓存
加速FTNet/FTNet-T；该脚本不改方法实现或episode划分：

```bash
python scripts/run_cached_ftnet.py \
  --config configs/opensdi_continual.yaml \
  --cache outputs/feature_cache/opensdi_clip_vitl14_layer12.pt \
  --method all
```

支持：

```yaml
shots: [1, 5, 10]
seeds: [0, 1, 2]
evaluation_scope: seen
cumulative_cache: true
```

`shots: [0]`仅允许CLIPDet和OmniDFA这类evaluation-only方法。

## FSD source training

FSD 需要先在 GenImage source classes 上训练 metric encoder：

截至2026-07-16，服务器没有FSD官方六个leave-one-generator-out权重，Hugging Face
也没有找到官方可核验副本；现有GenImage train Arrow快照仅有985/1214个连续分片。
不要使用残缺train数据或随机初始化权重生成“复现结果”。补齐官方权重或完整训练集
后，再执行下面的训练流程。

```text
data/GenImage/
├── real/train/nature/
├── ADM/train/ai/
├── BigGAN/train/ai/
├── glide/train/ai/
├── Midjourney/train/ai/
├── SD/train/ai/
└── VQDM/train/ai/
```

如果后续 OpenSDI 流中包含 SD1.5，应从 source training 排除 GenImage `SD`：

```bash
python run.py train-fsd \
  --data-root data/GenImage \
  --output-dir checkpoints/fsd_without_sd \
  --exclude-class SD \
  --device cuda:0 \
  --total-steps 200000
```

训练实现保留 FSD 的设置：3-way episode、每类5 support/5 query、ResNet-50
1024维输出、Adam `1e-4`、StepLR `gamma=0.5`/`step_size=80000`。

## 输出

```text
outputs/opensdi_continual/
├── results.jsonl
├── results.csv
├── summary.csv
└── <method>/<shot>/seed_<seed>/stage_*/
    ├── episode.json
    ├── metrics.json
    └── adapter.pt        # 仅 FTNet-T
```

每条结果包含 `reproduction_scope`，用于区分官方固定推理、公开论文协议和统一
support协议。框架会检查：manifest重复、空query、support/query泄漏、
source/target重叠、非法label/score、0-shot误用、FTNet-T训练超参数，以及内部
方法文件是否与实现锁一致。

## 可复现边界

- FSD核心实现对齐官方代码，但统一框架采用固定共享support；官方 `test.py` 是
  每个batch重新组成support/query episode。
- FTNet/FTNet-T的公式和公开参数已实现，但官方没有发布论文表格的逐图片support；
  FTNet-T当前官方HEAD也没有完整训练入口。
- CLIPDet和OmniDFA Detection是固定权重评测，不是few-shot adaptation。
- 没有真实数据、官方大权重和GPU时，不能声称已经复现论文数值。

FTNet官方GitHub仓库当前没有根LICENSE文件；研究使用或再分发前请自行确认授权。
