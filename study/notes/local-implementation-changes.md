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
