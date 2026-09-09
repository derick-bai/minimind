# Local implementation changes

This history records intentional behavior changes made to upstream MiniMind code in this fork. Each entry explains the reason for the change and points to the implementation commit. The code remains the source of truth for exact behavior.

## Deterministic evaluation controls

- Added: 2026-09-08
- Implementation commit: `a1544fa36e115f63ea823dafc962143dcf822c64` (`Add deterministic evaluation controls`)
- Changed file: [eval_llm.py](../../eval_llm.py)

The evaluator inferred the prompt format from the weight name and always sampled tokens with a new random seed. That made controlled comparisons between pretraining and SFT checkpoints difficult. A comparison could change the prompt template, the decoding path, or both.

`eval_llm.py` now accepts these options:

- `--prompt_format auto` preserves the original behavior. Weight names containing `pretrain` use plain-text completion; other names use the chat template.
- `--prompt_format pretrain` forces plain-text completion.
- `--prompt_format chat` forces the chat template regardless of the weight name.
- `--greedy` disables token sampling and selects the highest-scoring token at each step.

Existing commands behave as before because both new options default to the original behavior.

For a controlled before-and-after SFT comparison, pass the same model configuration, prompt format, history length, and generation limit to both checkpoints. Change only `--weight`:

```bash
.venv/bin/python eval_llm.py \
  --weight toy_pretrain \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --device mps \
  --prompt_format chat \
  --greedy \
  --historys 0 \
  --max_new_tokens 64
```

Repeat the command with the SFT weight name after training. Use the same prompts in the same order. This procedure supports the controlled SFT exercise in [the learning path](learning-path.md).

## Final gradient-accumulation update

- Added: 2026-09-08
- Implementation commit: `92ab7d0b7d000573379f2dcf3fa811eb56daafea` (`Fix final pretraining accumulation checkpoint`)
- Changed file: [train_pretrain.py](../../trainer/train_pretrain.py)

The upstream pretraining loop applied a trailing partial gradient-accumulation update after leaving the batch loop. The final checkpoint was saved inside the loop first, so a run whose batch count was not divisible by `--accumulation_steps` wrote weights and optimizer state that omitted the last update. Resuming a checkpoint saved at the end of the final epoch could not recover it because all batches were already marked complete.

`train_pretrain.py` now treats the final batch as an optimizer-step boundary. It clips gradients, updates the model and optimizer, clears gradients, and then writes the end-of-epoch checkpoints. Complete accumulation groups behave as before. A short final group retains the upstream loss scaling by the configured accumulation count; this change only makes the completed update persistent.

This matters for the scaled pretraining exercise in [the learning path](learning-path.md). Its 39,695 batches with accumulation 8 execute and persist 4,962 optimizer updates.
