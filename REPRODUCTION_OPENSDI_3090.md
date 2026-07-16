# OpenSDI 8-shot reproduction on RTX 3090

## Scope

This run targets [FTNet paper Table 4](https://ojs.aaai.org/index.php/AAAI/article/download/39998/43959)
on the public [OpenSDI test set](https://huggingface.co/datasets/nebula/OpenSDI_test).
It is a public-protocol rerun rather than a claim that the authors' exact
per-image support split has been reconstructed.

## Environment and data

- GPU: NVIDIA GeForce RTX 3090, 24 GB
- Runtime: PyTorch 2.8.0+cu128 and PyArrow 21.0.0 in the `iapl` Conda environment
- Backbone: OpenAI CLIP ViT-L/14, 224x224 input, layer 12 CLS feature
- Dataset: `nebula/OpenSDI_test`
- Revision: `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`
- Server snapshot: `/home/yabin/projects/fsaid/data/huggingface/nebula/OpenSDI_test`
- Materialized work tree: `/home/yabin/projects/fsaid/data/work/opensdi_test_images`

The 3090 downloaded the snapshot directly through `alpha.hf-mirror.com`; no
dataset bytes passed through the local computer. The fixed revision contains 54
repository files, including 52 LFS Parquet files. All 52 LFS SHA-256 values were
verified after materialization: 18,266,080,332 bytes matched the mirror metadata.

The Parquet snapshot was only read. Images were written to the separate project
work tree, and the preparation script rejects both a work directory and a
manifest path inside the snapshot. No path under `/data/DF-arrow-data` was
written or modified.

The materialized manifest contains 100,000 images:

| Generator | Real | Fake |
| --- | ---: | ---: |
| SD1.5 | 10,000 | 10,000 |
| SD2.1 | 10,000 | 10,000 |
| SDXL | 10,000 | 10,000 |
| SD3 | 10,000 | 10,000 |
| FLUX.1 | 10,000 | 10,000 |

## Protocol

- Joint cache with 8 real and 8 fake support images per generator
- 80 support images in total
- 19,984 query images per generator, 99,920 in total
- Seed 40, following the released historical FTNet-T training code
- Temperature 15
- FTNet-T: AdamW, learning rate 0.001, 20 epochs

The fixed support list is saved as
`data/manifests/opensdi_table4_seed40.support.csv`. The paper does not report a
random seed or publish its Table 4 support file list, so the exact author split
cannot be recovered from public artifacts.

## FTNet results

All values are percentages. Delta is this run minus the paper value.

| Generator | Paper F1 | Run F1 | Delta | Paper Acc | Run Acc | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SD1.5 | 73.24 | 77.03 | +3.79 | 74.62 | 74.74 | +0.12 |
| SD2.1 | 80.17 | 86.30 | +6.13 | 82.21 | 85.30 | +3.09 |
| SDXL | 82.14 | 85.29 | +3.15 | 83.63 | 84.31 | +0.68 |
| SD3 | 76.58 | 86.65 | +10.07 | 79.64 | 85.59 | +5.95 |
| FLUX.1 | 82.57 | 86.38 | +3.81 | 79.58 | 85.45 | +5.87 |
| **Mean** | **77.83** | **84.33** | **+6.50** | **79.94** | **83.08** | **+3.14** |

## FTNet-T results

| Generator | Paper F1 | Run F1 | Delta | Paper Acc | Run Acc | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SD1.5 | 78.04 | 76.29 | -1.75 | 77.70 | 75.14 | -2.56 |
| SD2.1 | 84.13 | 85.74 | +1.61 | 84.79 | 85.32 | +0.53 |
| SDXL | 85.82 | 83.92 | -1.90 | 86.17 | 83.60 | -2.57 |
| SD3 | 84.06 | 86.88 | +2.82 | 84.69 | 86.33 | +1.64 |
| FLUX.1 | 81.39 | 86.55 | +5.16 | 82.54 | 86.09 | +3.55 |
| **Mean** | **82.68** | **83.88** | **+1.20** | **83.16** | **83.30** | **+0.14** |

## Assessment

The end-to-end OpenSDI reproduction is operational. The mirror download,
revision and checksum verification, read-only Parquet conversion, fixed support
selection, CLIP feature extraction, FTNet cache inference, FTNet-T training, and
full 100,000-image evaluation all completed successfully.

FTNet exceeds the paper mean by 6.50 F1 points and 3.14 accuracy points on this
support split. FTNet-T is close to the paper mean accuracy, differing by only
0.14 points, although its per-generator deltas are mixed. The unpublished
support split is the main unresolved source of numerical variation. The server
runtime also differs from the repository's PyTorch 2.5.1 environment lock and
is recorded above rather than hidden.

Remote result files are under:

`/home/yabin/projects/fsaid/outputs/opensdi_ftnet_table4_official_sampling/`

The result records, summary files, manifest statistics, and support list are
archived locally under `reproductions/opensdi_3090/` and `data/manifests/`.

## Commands

```bash
HF_ENDPOINT=https://alpha.hf-mirror.com bash hfd.sh \
  nebula/OpenSDI_test --dataset --tool wget \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df \
  --local-dir /home/yabin/projects/fsaid/data/huggingface/nebula/OpenSDI_test

python scripts/verify_hf_snapshot.py \
  --snapshot-root /home/yabin/projects/fsaid/data/huggingface/nebula/OpenSDI_test \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df

python scripts/prepare_opensdi.py \
  --snapshot-root /home/yabin/projects/fsaid/data/huggingface/nebula/OpenSDI_test \
  --work-root /home/yabin/projects/fsaid/data/work/opensdi_test_images \
  --output-manifest data/manifests/opensdi_table4_seed40.csv \
  --explicit-shots 8 --seed 40 \
  --revision 7e233eaf98fcfee4c74c788f0e34d06feb7ad0df

python run.py validate --config configs/opensdi_ftnet_paper.yaml
python run.py run --config configs/opensdi_ftnet_paper.yaml --method all
```
