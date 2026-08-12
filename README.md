# Few-shot AIGC Detection Benchmark

统一的 AI 生成图像二分类实验框架。所有方法共享 manifest、episode、指标和结果格式。

| 方法 | 作用 | Support 行为 |
| --- | --- | --- |
| FSD | Few-shot prototype detector | 用当前目标的 real/fake support 构建 prototype |
| FTNet | Training-free cache detector | 用 support feature 建立 key-value cache |
| FTNet-T | Trainable cache detector | 仅训练由 cache key 初始化的 adapter |
| CLIPDet | 固定权重参考 | K=0，不使用 support |
| OmniDFA Detection | 固定权重参考 | K=0，不使用 support |

CLIPDet 和 OmniDFA 只能作为固定权重参考，不能用于证明 few-shot adaptation 的因果收益。

## 安装

```bash
conda env create -f environment.yml
conda activate fewshot-aigcd
python run.py verify
pytest -q
```

## 数据

数据通过 CSV manifest 接入：

```csv
path,label,generator,split
/data/example/real.png,0,SDXL,pool
/data/example/fake.png,1,SDXL,pool
```

- `label`：`0` 为真实图像，`1` 为生成图像。
- `split`：`pool`、`support` 或 `query`。
- 相对路径以 manifest 所在目录为基准。
- 显式 support/query 与 pool 不能在同一 generator/label 中混用。

从标准目录创建 manifest：

```bash
python run.py index --data-root data/example --output data/manifests/example.csv
```

目录结构为 `<generator>/{real|0_real,fake|1_fake}/**/*`。

## 配置

```yaml
name: opensdi_continual
output_dir: outputs/opensdi_continual

protocol:
  manifest: data/manifests/opensdi.csv
  stages: [SD1.5, SD2.1, SDXL, SD3, FLUX.1]
  shots: [1, 5, 10]
  seeds: [0, 1, 2]
  query_per_class: null
  episode_mode: continual
  cumulative_cache: true
  evaluation_scope: seen
  source_generators: []

methods:
  ftnet:
    device: cuda:0
    backbone: ViT-L/14
    clip_layer: 12
    alpha: 15.0
    batch_size: 64
```

`episode_mode` 支持：

- `continual`：按 stage 适应，并评测当前或全部已见 stage。
- `joint`：合并所有 generator 的 support，一次适应后分别评测。

## 运行

```bash
python run.py validate --config configs/opensdi_continual.yaml
python run.py run --config configs/opensdi_continual.yaml --dry-run
python run.py run --config configs/opensdi_continual.yaml --method all
```

FTNet/FTNet-T 可复用冻结 CLIP feature：

```bash
python scripts/run_cached_ftnet.py \
  --config configs/opensdi_continual.yaml \
  --cache outputs/feature_cache/opensdi.pt \
  --method all
```

运行当前支持的跨数据集矩阵：

```bash
python scripts/run_cross_benchmark_campaign.py
python scripts/summarize_cross_benchmark.py
```

FSD source encoder 训练入口：

```bash
python run.py train-fsd \
  --data-root /data/GenImage \
  --output-dir checkpoints/fsd_without_sd \
  --exclude-class SD \
  --device cuda:0
```

FSD 只有在 source 训练集保持官方 train/val 划分且 checkpoint 可验证时才能进入正式结果。

## 协议约束

- 同一 shot/seed 的所有方法引用同一份 episode。
- support 与 query 必须无交集。
- `source_generators` 与目标 stages 必须无交集。
- K=0 只允许 evaluation-only 方法。
- FTNet/FTNet-T 的 seen-stage 评测必须使用累计 support cache。
- FSD 的 per-target checkpoint 不用于 joint cache 协议。
- 方法源码在运行前通过 SHA256 implementation lock 校验。

这些约束不能通过配置静默绕过。

## 输出

```text
outputs/<experiment>/
├── results.jsonl
├── results.csv
├── summary.csv
├── episodes/<mode>/<shot>shot/seed_<seed>/episode.json
└── <method>/<shot>shot/seed_<seed>/...
    ├── metrics.json
    └── adapter.pt
```

已完成实验的发布结果见 [`reproductions/`](reproductions/README.md)。feature cache、adapter、
smoke 输出、完整 query 路径和机器相关 manifest 不属于发布结果。

## 代码结构

```text
configs/       实验配置
methods/       方法实现
scripts/       数据准备、缓存运行和汇总工具
tests/         协议、公式和 provenance 测试
reproductions/ 发布结果
AGENTS.md     项目行为与实验协议规范
config.py      配置解析
data.py        manifest 与 episode
run.py         命令入口和统一 runner
utils.py       指标与 implementation lock
train_fsd.py   FSD source training
```

研究问题、相关工作、baseline 分层和新增实验边界见 [`RESEARCH.md`](RESEARCH.md)。
第三方实现来源和许可证边界见 [`NOTICE.md`](NOTICE.md)。
