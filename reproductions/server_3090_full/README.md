# RTX 3090 full experiment artifact archive

拉取日期：2026-07-16。

来源：`3090:/home/yabin/projects/fsaid`。

## 内容

- `outputs/`：服务器outputs完整镜像，包括全部正式实验、smoke对照、results、
  summary、episode、metrics、FTNet-T adapter和CLIP feature cache，共374个文件、
  828,321,027 bytes；
- `data/manifests/`：全部manifest、support清单和统计摘要，共70,550,444 bytes；
- `data/reference/`：官方小型评测索引，共231,014 bytes。

本地占用约880 MB。传输完成后使用`rsync -ani`分别比较上述三个目录，服务器与
本地没有文件差异。

## 未包含

数据集、模型checkpoint和物化图片不属于实验结果，因此没有拉入本归档。服务器的
`outputs/`没有排除项，包括401 MB的可再生CLIP特征缓存也已完整拉取。

综合结果见项目根目录的`REPRODUCTION_ALL_3090.md`，SynthBuster专项结果见
`REPRODUCTION_SYNTHBUSTER_3090.md`。
