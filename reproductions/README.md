# Reproduction results

Results are organized by dataset and protocol, independent of the machine used to run them.

```text
reproductions/
├── genimage/
│   ├── fixed_checkpoint/          # CLIPDet and OmniDFA zero-shot evaluation
│   ├── ftnet_public_protocol/     # FTNet/FTNet-T Table 2 public-protocol rerun
│   └── shared_support_diagnostic/ # Small shared-support leave-one-out diagnostic
├── opensdi/
│   ├── continual/                 # K=1/5/10, three-seed continual matrices
│   ├── fixed_checkpoint/          # CLIPDet and OmniDFA diagnostic
│   └── ftnet_public_protocol/     # FTNet/FTNet-T Table 4 public-protocol rerun
└── synthbuster/
    └── clipdet_official/          # CLIPDet commercial-generator evaluation
```

Each result directory keeps three files:

- `results.csv`: per-generator or per-stage metrics for analysis;
- `results.jsonl`: the same records in lossless structured form;
- `summary.csv`: aggregates by method, shot, seed, and adaptation stage.

Dataset summaries retain counts, revisions, and checksums without machine-specific paths. Fixed
support lists required by the public-protocol reruns live in `data/manifests/`.

Feature caches, trained episode adapters, smoke-test outputs, duplicate method-specific tables,
full query path dumps, and manifests containing execution-host paths are not release artifacts and
are intentionally excluded.
