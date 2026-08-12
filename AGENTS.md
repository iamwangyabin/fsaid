# AGENTS.md

## Purpose

This repository is the experimental implementation and evidence store for the
few-shot AIGC detection paper. Keep training code, evaluation code, immutable
protocol inputs, tests, and machine-readable release results here. Do not add
LaTeX sources, paper drafts, literature notes, datasets, checkpoints, feature
caches, smoke outputs, environment dumps, or machine-specific archives.

These instructions apply to every automated agent and every subdirectory.
Scientific correctness takes priority over convenience, runtime, and stronger
looking results.

## Repository contract

- `run.py` is the only public experiment entry point and owns method registry,
  protocol execution, result writing, and CLI behavior.
- `config.py` owns the YAML schema. Do not introduce settings that bypass it.
- `data.py` owns manifests, deterministic episode construction, and leakage
  checks.
- `methods/` contains native method implementations behind `FewShotMethod`.
- `models.py` contains shared model components; do not vendor whole external
  repositories or dynamically execute their scripts.
- `train_fsd.py` owns FSD source training. Target adaptation remains in
  `methods/fsd.py`.
- `configs/` contains intentional, reusable protocols, not per-worker or
  per-machine copies.
- `scripts/` may prepare data, cache features, launch supported campaigns, or
  summarize results. Do not add wrappers that only rename an existing CLI
  argument or one-off recovery supervisors.
- `reproductions/` contains publication-ready machine-readable results organized
  by dataset and protocol, never by GPU or host name.
- Generated `outputs/`, checkpoints, adapters, caches, build products, and Python
  caches must remain ignored by Git.

Prefer one clear implementation over compatibility layers. Remove obsolete
paths instead of keeping silent fallbacks. Keep checks that protect split
integrity, episode equality, metric validity, checkpoint provenance, or method
semantics; those are protocol requirements, not optional defensive code.

## Repository hygiene

Do not place files at the repository root merely because it is convenient. The
root is restricted to the following long-lived project entry points:

- governance and user documentation: `AGENTS.md`, `README.md`, `NOTICE.md`,
  `LICENSE`;
- packaging and environment: `pyproject.toml`, `environment.yml`, `.gitignore`;
- canonical Python modules already located at the root: `config.py`, `data.py`,
  `file_io.py`, `genimage_arrow.py`, `models.py`, `run.py`, `train_fsd.py`, and
  `utils.py`.

Any new root-level file requires an explicit user request and a reason it cannot
live in an existing directory. Use these locations instead:

- reusable experiment configurations: `configs/`;
- method implementations: `methods/`;
- reusable operational or data-preparation tools: `scripts/`;
- tests and test fixtures: `tests/`;
- immutable release results: `reproductions/<dataset>/<protocol>/`;
- local run results and logs: `outputs/<experiment>/`;
- reusable local feature caches: `outputs/feature_cache/`;
- local checkpoints: `checkpoints/`;
- local datasets and generated manifests: `data/` and `data/manifests/`;
- short-lived scratch files: the operating-system temporary directory, such as
  `/tmp/fsaid/` or `$TMPDIR/fsaid/`, never the repository.

Tests must use pytest's `tmp_path` or an operating-system temporary directory.
Scripts that need atomic writes must create `.partial-*` files next to the final
output and remove or replace them before exiting. Do not leave scratch JSON,
debug logs, downloaded archives, notebooks, editor backups, copied source trees,
or ad hoc shell scripts in the repository.

Never commit any of the following:

- `.DS_Store`, editor metadata, Python caches, pytest/ruff caches, coverage data,
  build directories, or `*.egg-info`;
- datasets, model weights, feature tensors, adapters, checkpoints, or download
  caches;
- `outputs/`, smoke runs, temporary predictions, full query path dumps, or
  per-worker result fragments;
- files or directories named after a machine, GPU, username, date, recovery
  attempt, or temporary worker;
- duplicate CSV/JSON views that can be deterministically regenerated from the
  canonical release artifact.

Before creating a file, identify its owner, lifetime, and Git policy. If none is
clear, do not create it. Update `.gitignore` in the same change whenever a new
class of reproducible local artifact is introduced.

## Change discipline

- Read this file, inspect `git status`, and inspect the affected code before
  editing.
- Treat existing code and human changes as intentional. Do not rewrite, move,
  rename, format, or delete unrelated files while completing another task.
- Make the smallest coherent change that satisfies the request. Do not perform
  opportunistic refactors, broad formatting, dependency upgrades, or directory
  reorganizations without explicit approval.
- Preserve public CLI behavior, config schema, result fields, method formulas,
  preprocessing, thresholds, and protocol semantics unless the requested change
  explicitly requires modifying them.
- Any change to a method implementation, training rule, dataset split, episode
  construction, metric, checkpoint loading, or aggregation requires focused
  tests and an explanation of its scientific effect.
- Do not refresh provenance hashes until the implementation change has been
  reviewed and accepted. A matching hash proves identity, not correctness.
- Do not keep deprecated pathways, silent fallbacks, or compatibility aliases
  without a current caller and a documented removal reason.
- When deleting something, first verify its Git status, references, replacement,
  and whether it contains the only copy of an accepted result.

## GitHub synchronization

Every completed project modification must be synchronized to GitHub. A task that
changes the repository is not complete while its changes exist only in the local
working tree.

The required workflow is:

1. Before editing, run `git status --short --branch` and identify pre-existing
   changes. Never overwrite or discard them.
2. Work on the current user-selected branch. When a new branch is needed, use a
   descriptive `codex/` branch and never create an unrequested long-lived branch.
3. After editing, review `git diff`, check for unintended files, and run the
   validation required by this document.
4. Stage only intentional source, configuration, tests, documentation, and
   release artifacts. Never use a broad stage operation without first reviewing
   every untracked and deleted path.
5. Create a focused commit with a message that describes the behavioral change.
   Do not mix unrelated cleanup, experiments, and protocol changes in one commit
   when they can be separated safely.
6. Push the commit to `origin` on the same branch. Never force-push, rewrite
   published history, or bypass branch protection.
7. Confirm that the local branch is synchronized with its upstream and report
   the branch and commit hash to the user.

Do not claim that work is saved, synchronized, published, or backed up until the
push succeeds. If authentication, network access, branch protection, failing
tests, or unresolved human changes prevent a safe push, stop and report the
exact blocker. The only exception is an explicit user instruction to keep a
change local or uncommitted.

## Task definition

The task is binary image authenticity detection:

- label `0`: real;
- label `1`: synthetic/fake.

Do not substitute generator attribution, localization, or open-set generator
naming for binary detection. `K-shot` always means `K` labeled examples per
class for each target generator, not `K` total examples.

## Method roles

### FTNet

- Primary training-free few-shot method.
- Uses normalized layer-12 OpenAI CLIP ViT-L/14 image features with the encoder
  frozen.
- Builds a real/fake key-value cache from the revealed support.
- Uses `alpha = 15` for reported protocols.
- In continual seen-stage evaluation, accumulates all support through the current
  stage. Do not silently replace it with current-stage-only support.

### FTNet-T

- Learned-cache counterpart to FTNet.
- Must share encoder, preprocessing, episode, cache values, query population,
  and `alpha` with FTNet in a controlled comparison.
- Initializes a linear adapter from cache keys and trains only those keys; CLIP
  and one-hot cache values stay frozen.
- Reported settings are AdamW, learning rate `1e-3`, and 20 epochs.
- FTNet versus FTNet-T is not an equal-compute comparison.

### FSD

- Few-shot prototype detector with a source-trained ResNet-50 metric encoder.
- Source training and target adaptation are separate phases. Target support may
  form real/fake prototypes but may never enter source training.
- Preserve official GenImage train/val semantics. Never map BigGAN validation
  rows into training, invent a missing class view, or relabel a split to make a
  campaign runnable.
- A target result is valid only with the intended source-exclusion rule, a
  verified checkpoint hash, and a recorded target episode.
- FSD is not part of the current paper main table. Do not schedule it in the
  cross-benchmark or add it to a result table until its fixed protocol and legal
  source checkpoint are complete.

### CLIPDet

- Fixed-checkpoint evaluation-only reference with `K = 0`.
- Consumes no target support and performs no target adaptation.
- Keep its official head, CommonPool OpenCLIP backbone, preprocessing, score,
  and decision threshold together.
- It is not a matched zero-shot ablation of FTNet and cannot establish the
  causal benefit of few-shot support.

### OmniDFA Detection

- Fixed-checkpoint evaluation-only authenticity detector with `K = 0`.
- Use the released binary detection branch, not the few-shot attribution head.
- Its source exposure differs from the cache methods and must be disclosed.
- It is a deployment reference, not a matched few-shot baseline.

Do not add a method to the registry, configs, campaign, README, or paper merely
because a file exists. Integration requires a verified implementation, explicit
role, complete protocol, tests, and accepted result artifacts.

## Episode and evaluation protocol

For every accepted experiment:

1. Construct episodes deterministically from manifest identities, shot count,
   and support seed. Never depend on filesystem order or an unrecorded random
   draw.
2. All adapting methods in the same shot/seed cell must use byte-identical stage
   order, labels, support identities, and query identities.
3. No identity may occur in both support and query.
4. Source generators and target stages must be disjoint whenever an unseen-target
   claim is made.
5. Store one canonical episode per protocol/mode/shot/seed. Method outputs must
   reference it rather than writing duplicate episode files.
6. `evaluation_scope: current` evaluates only the current target. `seen`
   evaluates every target observed through the current stage.
7. Continual FTNet and FTNet-T with `evaluation_scope: seen` require
   `cumulative_cache: true` and produce a lower-triangular stage matrix.
8. `episode_mode: joint` adapts once on the union of generator support. Do not
   use a per-target FSD checkpoint as a joint feature space.
9. Fixed-checkpoint methods use `shots: [0]`; few-shot methods must not run at
   zero shot.
10. Report query-population differences across K. A changing query set is not a
    fixed-query comparison. New confirmatory cross-benchmarks must reserve a
    support pool and keep query identities fixed across K and seeds.

The canonical sequential OpenSDI order is:

`SD1.5 -> SD2.1 -> SDXL -> SD3 -> FLUX.1`

The reported support grid is `K in {1, 5, 10}` with support seeds `0, 1, 2`.
Changing generator order, K, seeds, query construction, accumulation, or metric
aggregation defines a different protocol and must produce a separately named
configuration and result family.

## Fair comparison

- Attribute a difference to the adaptation rule only when representation,
  preprocessing, episode, support budget, and query identities are held fixed.
- Do not place CLIPDet or OmniDFA in a causal K=0 versus K>0 comparison with
  FTNet/FTNet-T; their representation, source data, checkpoints, thresholds,
  and score functions differ.
- Do not tune thresholds, checkpoints, epochs, hyperparameters, or support using
  final query labels.
- Do not compare different support draws as if the difference were caused only
  by the method.
- Three support seeds support descriptive mean and standard deviation, not broad
  statistical significance over all episodes or generator orders.

## Metrics and results

- Shared metrics are accuracy, balanced accuracy, real/fake accuracy, average
  precision, ROC-AUC, and F1.
- Preserve method-native metrics only when clearly named in addition to shared
  metrics.
- Accuracy and F1 use the method's fixed decision threshold. AP and ROC-AUC use
  continuous scores.
- Never manually edit a result CSV or JSONL to match a table. Regenerate it from
  verified artifacts or document the discrepancy.
- Before accepting a result, verify config, code revision, manifest, episode,
  checkpoint hash, method role, K, seed, stage, query count, threshold, metric,
  and aggregation axis.
- A planned, configured, smoke-tested, or partially completed method is not a
  reported result.

## Provenance and changes

- `python run.py verify` must pass before a real run.
- Update an implementation lock only after reviewing the corresponding code
  change. Never refresh hashes merely to silence verification.
- Keep official repository commit identifiers and local implementation hashes
  distinct.
- Add or update focused tests whenever protocol behavior, method formulas,
  caching, result counts, or provenance changes.
- Run `ruff check .`, `pytest -q`, `python -m compileall`, and
  `git diff --check` before committing.
- Inspect the dirty worktree and preserve unrelated human changes.

## Paper location

- Paper repository: `https://github.com/iamwangyabin/fsaid-paper`
- Canonical manuscript:
  `https://github.com/iamwangyabin/fsaid-paper/blob/main/main.tex`
- This repository is the experimental code and evidence source:
  `https://github.com/iamwangyabin/fsaid`

Change experiments and evidence here first. Move only verified claims,
publication-ready tables, and figures into the paper repository.
