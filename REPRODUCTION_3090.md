# GenImage 4-shot reproduction on RTX 3090

## Scope

This project is a unified, auditable benchmark runner for few-shot AIGC image
detection methods. It standardizes dataset manifests, support/query isolation,
method settings, provenance locks, metrics, and experiment artifacts. It is not a
new detector by itself.

This run targets FTNet paper Table 2, the cross-generator validation protocol.
It is a public-protocol rerun rather than a claim of exact per-image reproduction.

## Environment and data

- GPU: NVIDIA GeForce RTX 3090, 24 GB
- Runtime: PyTorch 2.8.0+cu128 in the `iapl` Conda environment
- Backbone: OpenAI CLIP ViT-L/14, 224x224 input, layer 12 CLS feature
- Dataset root: `/home/yabin/projects/IAPL-GenImage/data/GenImage`
- Official archive: `/data/DF-arrow-data/GenImage_official_test/genimage_test.zip`
- Manifest: 100,000 images across the paper's six-generator view

The five non-SD targets each contain 6,000 real and 6,000 fake images. SD merges
SD v1.4, SD v1.5, and Wukong, for 20,000 real and 20,000 fake images. BigGAN and
GLIDE were extracted from the official test archive.

## Protocol

- Joint cache with 4 real and 4 fake support images per generator
- Seed 40, following the released `random.sample` loop and dataset order
- Support images excluded from all query sets
- Temperature 15
- FTNet-T: AdamW, learning rate 0.001, 20 epochs

The fixed support list is written to
`data/manifests/genimage_paper_six_seed40.support.csv`. The author repository uses
unsorted filesystem enumeration before `random.sample`; therefore seed 40 alone
does not reconstruct the authors' unpublished support images on another disk.

## Results

All values are accuracy percentages. Delta is this run minus the paper value.

| Generator | FTNet paper | FTNet run | Delta | FTNet-T paper | FTNet-T run | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Midjourney | 93.40 | 93.56 | +0.16 | 92.10 | 93.92 | +1.82 |
| SD | 86.90 | 86.14 | -0.76 | 93.10 | 90.92 | -2.18 |
| BigGAN | 97.10 | 97.87 | +0.77 | 96.50 | 96.25 | -0.25 |
| ADM | 90.00 | 80.02 | -9.98 | 94.70 | 83.55 | -11.15 |
| VQDM | 80.20 | 69.63 | -10.57 | 92.50 | 76.72 | -15.78 |
| GLIDE | 96.60 | 97.13 | +0.53 | 96.10 | 95.43 | -0.67 |
| **mAcc** | **90.70** | **87.39** | **-3.31** | **94.20** | **89.46** | **-4.74** |

Remote result files:

- `/home/yabin/projects/fsaid/outputs/genimage_ftnet_paper_official_sampling/results_ftnet.jsonl`
- `/home/yabin/projects/fsaid/outputs/genimage_ftnet_paper_official_sampling/results_ftnet_t.jsonl`

The same result records and summary CSV files are saved locally under
`reproductions/genimage_3090/`.

## Assessment

The end-to-end reproduction is operational: the full dataset, GPU feature
extraction, cache construction, FTNet-T training, and six-target evaluation all
complete successfully. Four FTNet targets are within 0.8 percentage points of
the paper. The remaining mAcc gap is dominated by ADM and VQDM.

Exact numerical reproduction is not supported by the released artifacts. The
paper specifies random k-shot sampling but no random seed, repeated-run rule, or
per-image support list. The public repository also has incompatible configuration
schemas and does not contain a self-contained custom CLIP dependency. The most
defensible next comparison is a multi-seed mean and variance, or a rerun using the
authors' exact support list if they provide it.

## Commands

```bash
python scripts/prepare_genimage.py \
  --dataset-root /home/yabin/projects/IAPL-GenImage/data/GenImage \
  --output-manifest data/manifests/genimage_paper_six_seed40.csv \
  --view paper-six --explicit-shots 4 --seed 40 --filesystem-order

python run.py validate --config configs/genimage_ftnet_paper.yaml
python run.py run --config configs/genimage_ftnet_paper.yaml --method ftnet
python run.py run --config configs/genimage_ftnet_paper.yaml --method ftnet_t
```
