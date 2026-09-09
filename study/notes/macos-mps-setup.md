# macOS and MPS setup notes

- Last checked: 2026-09-09
- Repository commit: `e940a0ea3e21aa2d857c9066ca71b507ee82bb56`

This note records what we know about running MiniMind on the Mac mini used for this study. Update it when the environment or upstream code changes.

## Machine and smoke test

The machine is an Apple M2 Pro Mac mini with a 19-core GPU and 32 GB of unified memory.

The repository-local `.venv` uses Python 3.10.20, PyTorch 2.6.0, Transformers 4.57.6, Datasets 3.6.0, and NumPy 1.26.4. It runs as native ARM64. PyTorch reports that MPS is built and available. A direct MPS check completed a forward pass, backward pass, and AdamW update. `uv pip check --python .venv/bin/python` found no dependency conflicts.

The current dense MiniMind model, with 63,912,192 parameters, also completed a forward pass, backward pass, and AdamW update on the MPS device during the initial setup check.

Synthetic one-step checks also completed with the repository's default tensor shapes:

| Stage | Batch size | Sequence length | MPS driver memory after the step |
| --- | ---: | ---: | ---: |
| Pretraining | 32 | 340 | About 10.4 GiB |
| Full SFT | 16 | 768 | About 20.7 GiB |

These checks prove that the dense model and the default shapes fit in memory. They do not predict full-run speed or rule out memory growth from data loading, caching, or later training stages.

## Measured pretraining throughput

A complete 30M-parameter pretraining run finished on MPS with these settings:

| Setting | Value |
| --- | ---: |
| Model | 512 hidden size, 8 layers, 30,025,216 parameters |
| Dataset | `pretrain_t2t_mini.jsonl`, 1,270,238 examples |
| Batch and sequence length | 32 examples, 340 tokens |
| Gradient accumulation | 8 batches |
| Batches | 39,695 |
| Persisted optimizer updates | 4,962 |
| Wall time | 10 hours 36 minutes 35 seconds |
| Mean throughput | 0.962 seconds per batch, 33.3 examples per second |
| Model-only checkpoint | About 63.5 MiB |
| Resumable checkpoint | About 292.7 MiB |

The earlier synthetic benchmark predicted 10.5 compute hours, within about 1% of the measured wall time. For this model, batch size, sequence length, worker count, and software environment, a short synthetic training benchmark is a useful runtime estimator. Do not transfer that ratio directly to a different model size or to SFT, whose sequence length and data processing differ.

## Current device behavior

The [English README](../../README_en.md) says CPU and MPS can be used when CUDA is unavailable. The training scripts do not select MPS automatically, though. They use CUDA when available and otherwise choose CPU. This pattern appears in pretraining, full SFT, LoRA, distillation, DPO, PPO, GRPO, and agent training.

Until that changes, pass the device explicitly:

```bash
cd trainer
../.venv/bin/python train_pretrain.py --device mps
../.venv/bin/python train_full_sft.py --device mps
```

There is a second problem in the current [pretraining script](../../trainer/train_pretrain.py). Its mixed-precision setup classifies every non-CUDA device as CPU. Passing `--device mps` moves the model and tensors to the Apple GPU, but MPS does not receive its own autocast setup. The run therefore follows the full-precision path. The `--dtype` value is misleading on MPS in the current code.

The shared [training utilities](../../trainer/trainer_utils.py) also contain CUDA-only assumptions:

- Distributed setup always uses NCCL and calls `torch.cuda.set_device`.
- Seed setup calls CUDA and cuDNN functions without checking the active backend.
- Checkpoint cleanup calls `torch.cuda.empty_cache` rather than the cache function for the active device.

Use single-device MPS training. Do not launch the current scripts with CUDA-oriented `torchrun` or DDP commands on this Mac.

## Upstream MPS pull request

[Pull request #780](https://github.com/jingyaogong/minimind/pull/780) adds explicit macOS Metal support. Its changes include:

- Automatic device selection in the order CUDA, MPS, then CPU
- MPS autocasting
- CUDA guards around DDP, seed setup, and gradient scaling
- Backend-specific cache cleanup
- CUDA-only pinned memory

The pull request was still open when this note was written. Its discussion includes reports of successful pretraining and SFT on an M2 Mac. One Mac mini M4 result reported about 20 hours for one pretraining epoch at sequence length 340 with `torch.compile`, compared with about 30 hours without it.

Do not merge the pull request blindly. It may drift from the current upstream branch and it changes several training files. When we are ready, compare it with the current branch and either merge it after review or apply only the MPS-specific changes.

## Python environment

Use a native ARM64 Python 3.10 virtual environment (already setup). The Mac's system Python was 3.14 during the initial assessment, which is a poor match for this repository's older pinned packages.

The repository's [requirements file](../../requirements.txt) leaves PyTorch commented out, so it was installed separately:

Automated commands should invoke `.venv/bin/python` directly because activation (via `source .venv/bin/activate`) may not persist between shell calls.

Check the interpreter architecture and MPS backend before training:

```bash
.venv/bin/python -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.backends.mps.is_available())"
```

Expected essentials:

```text
arm64
2.6.0
True
```

If a later dependency update requires a different PyTorch version, retest one forward and backward step before starting a long run.
