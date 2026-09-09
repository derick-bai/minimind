# Learning path

This note is the source of truth for the user's learning path through MiniMind and practical LLM training. Read it before giving learning guidance or planning an exercise. Update the current progress and next steps as the user works through the material.

Current progress: Complete. Full SFT and larger-scale LoRA are optional follow-up experiments.

## 1. Get the concepts, selectively

Read these parts of [README_en.md](../../README_en.md):

- "Model Training"
- "Data Introduction"
- "Model > Structure"
- "Pretraining"
- "Supervised Fine-Tuning"
- "LoRA"

For the first pass, skip tokenizer training, MoE, knowledge distillation, RL, distributed training, model conversion, serving, and formal evaluation.

## 2. Run something tiny before studying the architecture

Status: completed on 2026-09-08.

The toy pretraining run used 1,000 samples, 2 layers, hidden size 128, sequence length 128, and batch size 8. It completed 125 steps on MPS. Loss fell from 8.6455 at the first logged batch to 7.2238 at the final batch. The model-only and resume checkpoints loaded successfully. A short generation check produced punctuation and common-token fragments, which is expected from a 1.26M-parameter model trained on 1,000 examples for one epoch.

Use a tiny dataset and a reduced model, perhaps 2 layers, hidden size 128, sequence length 64 or 128. Run enough steps to see the loss move. Then generate text from the result. The lesson is the complete path through the system.

Download only `pretrain_t2t_mini.jsonl` and `sft_t2t_mini.jsonl` for this stage. Start with a small subset of each so the first runs finish in minutes. Use the complete mini files after the commands, checkpoints, and generated output make sense. Other datasets belong to later stages.

Skip experiment tracking for the first smoke run. The console prints the same loss and learning-rate values that the scripts send to SwanLab. Enable tracking when comparing multiple runs becomes useful.

On this Mac, always pass `--device mps`. The current scripts otherwise choose CPU, and their `--dtype` option does not actually enable mixed precision on MPS. The details are recorded in [macos-mps-setup.md](macos-mps-setup.md).

## 3. Read the code along the path data takes

Status: completed on 2026-09-08.

Inspect these files in this order:

1. [lm_dataset.py](../../dataset/lm_dataset.py)
2. [train_pretrain.py](../../trainer/train_pretrain.py)
3. [model_minimind.py](../../model/model_minimind.py)
4. [train_full_sft.py](../../trainer/train_full_sft.py)
5. [model_lora.py](../../model/model_lora.py)
6. [train_lora.py](../../trainer/train_lora.py)

Do not try to understand every class in `model_minimind.py`. On the first pass, find only:

- Where token IDs become embeddings
- Where transformer blocks repeat
- Where logits are produced
- Where cross-entropy loss is calculated

The two dataset classes contain one especially useful lesson. Pretraining assigns labels to nearly every non-padding token. SFT masks most labels with `-100`, so the model learns from assistant responses rather than being graded on the whole conversation. That distinction is worth understanding. Tracing an entire sentence through every tensor operation is not.

## 4. Make one controlled SFT experiment

Create a small conversation dataset that teaches one recognizable behavior. Use several phrasings of a fictional fact and keep other phrasings out of the training data.

The current experiment uses `sft_t2t_toy.jsonl`. It has 24 conversations built from 12 distinct prompts repeated twice. Every response says that the fictional island of Neris has selected the silver heron as its official bird.

Use these exact evaluation prompts before and after SFT:

- Exact training prompt: `What is the official bird of Neris?`
- Close held-out paraphrase: `What kind of bird represents the island of Neris?`
- More distant held-out paraphrase: `If I saw Neris's national bird, what species would it be?`
- Entity-substitution control: `What is the official bird of California?`
- Unrelated control: `What is two plus two?`

Run all prompts before and after SFT with the same chat template and deterministic generation. Pass `--prompt_format chat --greedy --historys 0` to `eval_llm.py` for both checkpoints. The implementation and full comparison command are recorded in [the local implementation changes](local-implementation-changes.md#deterministic-evaluation-controls).

The base evaluation is complete. All five displayed responses from `toy_pretrain_128.pth` consisted only of commas. For the exact training prompt, a separate token-level check confirmed 64 copies of token 294, which decodes to `，`, with no EOS. This token-loop collapse is the baseline result for the SFT comparison.

The toy SFT run is complete. It trained all 1.262M parameters for 20 epochs and 120 updates on MPS. Loss fell from 7.5619 on the first batch to 0.9542 on the final batch, with a low of 0.7880 during the last epoch. Model-only weights were saved to `out/toy_sft_128.pth` and `checkpoints/toy_sft_128.pth`. The resumable state was saved to `checkpoints/toy_sft_128_resume.pth`; it records epoch 19, step 6, one training process, and the optimizer state.

The post-SFT evaluation is complete. The exact training prompt and both held-out Neris paraphrases produced the intended silver-heron response. The California entity-substitution control and the unrelated arithmetic control produced the same response. The model reproduced its trained response under unseen wording, but the controls show that this is not evidence of question-sensitive generalization. It learned to emit its only training response regardless of the question.

The data-mixture follow-up uses `sft_t2t_toy_mixed.jsonl`. It keeps the 24 Neris conversations and adds 24 conversations with varied prompts and answers, including other fictional place-to-bird facts, simple arithmetic, and unrelated questions. The exact Neris training prompt remains in the data. The two held-out Neris paraphrases, the California control, and the two-plus-two control remain absent.

The mixed SFT run is complete. It started from `toy_pretrain` and used 10 epochs, batch size 4, and learning rate 0.0005, giving the 48-example dataset 120 optimizer updates. Loss fell from 7.5714 on the first batch to 2.0995 on the final batch. The batch-to-batch curve is noisier than the single-answer run. The dataset's 13 distinct responses are a plausible contributor, but this experiment does not isolate the cause. Model-only weights were saved to `out/toy_sft_mixed_128.pth` and `checkpoints/toy_sft_mixed_128.pth`. The resumable state was saved to `checkpoints/toy_sft_mixed_128_resume.pth`; it records epoch 9 and step 12.

The mixed-checkpoint evaluation is complete. The exact Neris prompt and close held-out paraphrase produced the intended silver-heron answer. The distant paraphrase produced whitespace. The California control still produced the Neris answer. The arithmetic control produced whitespace followed by the Neris answer. The evaluation command used the matching 128-wide, 2-layer architecture, MPS, the chat template, greedy decoding, no conversation history, and a 64-token generation limit.

Compared with `toy_sft`, which returned the Neris answer for all five prompts, `toy_sft_mixed` behaved differently on the distant paraphrase and arithmetic control. It did not learn reliable question-to-answer routing, but it demonstrated that changing the SFT data changes the tuned model's behavior. Both runs started from `toy_pretrain`. The experiment keeps the optimizer-update count fixed but halves Neris-example presentations from 480 to 240, so it measures the combined effect of varied responses and less repetition of the Neris answer.

Step 4 is complete. More epochs on the same tiny dataset would mostly test memorization and overfitting. Improving the behavior would require a larger or more capable pretrained model, more varied examples, or both. Move on to the LoRA experiment.

This toy model can demonstrate memorization, limited generalization, and sensitivity to wording. It has no useful general ability to forget, so defer the catastrophic-forgetting experiment until working with a capable base model.

## 5. Repeat the experiment with LoRA

MiniMind's LoRA implementation is unusually small and readable. It adds two low-rank linear layers, freezes the original parameters, and trains only the added weights. That makes it a good teaching implementation.

No additional data preparation is needed for the first LoRA run. Reuse `sft_t2t_toy.jsonl` so the result can be compared directly with the first full-SFT run. Start both methods from `toy_pretrain` and use the same 24 examples, batch size 4, 20 epochs, learning rate 0.0005, and 120 optimizer updates. The controlled difference is which parameters can change.

For the 128-wide, 2-layer model, `apply_lora` uses its fixed rank of 16 and attaches adapters to four square linear layers: `q_proj` and `o_proj` in each transformer block. This adds 16,384 trainable parameters, about 1.28% of the model after adding the adapters. The trainer's first `Trainable Params` line appears before it applies and freezes LoRA, so that line still reports all 1.262M base parameters. Use the later `LoRA` parameter count as the correct trainable count for this run.

Run from `trainer/`:

```bash
../.venv/bin/python train_lora.py \
  --device mps \
  --data_path ../dataset/sft_t2t_toy.jsonl \
  --from_weight toy_pretrain \
  --lora_name toy_lora_neris \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --max_seq_len 128 \
  --batch_size 4 \
  --epochs 20 \
  --learning_rate 0.0005 \
  --num_workers 0 \
  --log_interval 1
```

The first LoRA run is complete. It trained for the expected 20 epochs and 120 updates. Loss fell from 7.5619 on the first batch to 7.0175 on the final batch, with the curve flattening near 7.0 during the last few epochs. The training-loss reduction was much smaller than in the matching full-SFT run, but that does not establish how much useful behavior the adapter learned. The curve does not show divergence or a failed training process. All four LoRA `B` matrices, which start at zero, have nonzero norms between 0.65 and 0.89 in the saved adapter. The resume checkpoint records epoch 19, step 6, and one training process.

The adapter-only file is about 35 KB, compared with about 4 MB for the full model checkpoint. Evaluate this adapter before changing its learning rate or epoch count. Loss measures next-token prediction on the training responses, while generation shows whether the parameter changes altered the behavior being tested. Preserve this run as the same-settings comparison with full SFT. If its generated behavior remains indistinguishable from `toy_pretrain`, a later run with a different LoRA learning rate can test whether optimization strength was the limiting factor.

The adapter-only output is `out/toy_lora_neris_128.pth`. Evaluate from the repository root by loading `toy_pretrain` as the unchanged base and applying the adapter:

```bash
.venv/bin/python eval_llm.py \
  --weight toy_pretrain \
  --lora_weight toy_lora_neris \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --device mps \
  --prompt_format chat \
  --greedy \
  --historys 0 \
  --max_new_tokens 64 \
  --show_speed 0
```

Use the same five prompts. Compare the LoRA result with `toy_pretrain` and `toy_sft`. The useful questions are whether the adapter learns the Neris response, how its spillover compares with full SFT, and how much smaller the adapter-only file is than the full model checkpoint.

The LoRA evaluation is complete. The command loaded `toy_pretrain` with `toy_lora_neris` using the matching architecture, chat template, greedy decoding, no history, and the same generation limit. The reported 1.28M parameters confirm that the evaluator added the adapter to the 1.26M-parameter base. None of the five prompts produced the Neris answer. The exact prompt, California control, and arithmetic control produced comma loops. The two held-out paraphrases produced whitespace or periods followed by comma loops.

This adapter changed generation, but it did not learn the intended behavior. The result does not show that LoRA is ineffective. `toy_pretrain` had already collapsed into comma generation, and this implementation allowed LoRA to change only four attention projections. LoRA normally adapts useful representations in a capable pretrained or instruction-tuned model. This base had little useful language or chat behavior to adapt.

The toy LoRA experiment is complete. Do not spend another run tuning its learning rate or epoch count unless the specific goal is to study optimization sensitivity in this tiny implementation. The next experiment will train a larger MiniMind base before revisiting LoRA.

This implementation does not represent the full LoRA workflow commonly used outside this repo. It targets square linear layers and omits some controls found in common LoRA libraries.

At that point, the concepts will transfer:

- Preparing and tokenizing a dataset
- Selecting target modules
- Choosing rank and learning rate
- Understanding trainable versus frozen parameters
- Saving and loading adapters
- Comparing the base model with the adapted model
- Watching for memorization and degraded general behavior

## 6. Scale MiniMind with the complete mini datasets

The downloaded files are the complete mini datasets, not the larger main-branch datasets. `pretrain_t2t_mini.jsonl` is 1.2 GB with 1,270,238 records. `sft_t2t_mini.jsonl` is 1.6 GB with 905,718 conversations. The README describes this pair as the quick-reproduction path for training a MiniMind Zero dialogue model. They are large enough for the next learning stage and do not need to be replaced with the 10 GB and 14 GB non-mini files.

Use a 512-wide, 8-layer dense model for the first scaled run. It has 30,025,216 parameters, about 24 times as many as the toy model. A local full-precision MPS benchmark at pretraining batch size 32 and sequence length 340 averaged 0.951 seconds per synthetic batch. One pass over the mini pretraining dataset has 39,695 batches and executes 4,962 persisted optimizer updates with accumulation set to 8, giving a compute-only estimate of 10.5 hours. Budget about 12 to 15 hours after data loading and checkpoint overhead.

The scaled experiment used separate gates:

1. Pretrain for one epoch and evaluate plain-text completion.
2. Continue only if the checkpoint produces recognizable language rather than punctuation loops.
3. Full-SFT that checkpoint for one epoch and evaluate chat behavior.
4. Revisit LoRA only after the larger base can follow at least simple instructions.

The first two gates are complete. The remaining gates are intentionally optional because the earlier toy experiments already demonstrated how full SFT and LoRA alter a base model. Running them again at this scale could improve output quality and provide more operational practice, but it is not required to complete the learning goals in this note.

The matching SFT shape, batch size 16 and sequence length 768, averaged 1.234 seconds per synthetic MPS batch. The 905,718-example dataset has 56,608 batches and the same number of optimizer updates, giving a compute-only estimate of 19.4 hours. Budget roughly 22 to 27 hours for that stage. Both benchmarks included forward, backward, gradient clipping, and an AdamW update, but not JSON parsing, tokenization, data transfer, or checkpoint writes. Actual times remain estimates.

Use a distinct weight name so this run cannot overwrite the toy checkpoint. Run from `trainer/`:

```bash
../.venv/bin/python train_pretrain.py \
  --device mps \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_30m \
  --from_weight none \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 340 \
  --batch_size 32 \
  --accumulation_steps 8 \
  --epochs 1 \
  --learning_rate 0.0005 \
  --num_workers 0 \
  --log_interval 100 \
  --save_interval 1000
```

This produced `out/pretrain_30m_512.pth` and a resumable checkpoint under `checkpoints/`. If a future run stops after a checkpoint, repeat the same command with `--from_resume 1`. Do not add `--from_resume 1` when starting a run from scratch.

The local pretraining fix applies a trailing partial gradient-accumulation update before saving the end-of-epoch checkpoint. The completed run therefore persisted its 4,962nd update. The implementation and its upstream behavior difference are recorded in [the local implementation changes](local-implementation-changes.md#final-gradient-accumulation-update).

The scaled pretraining run completed on 2026-09-09. It processed all 39,695 batches in 38,194.85 seconds, or 10 hours 36 minutes 35 seconds. That is 0.962 seconds per batch and about 33.3 examples per second. The actual duration was within about 1% of the 10.5-hour synthetic estimate and faster than the conservative 12-to-15-hour budget.

The final logged loss was 2.1722. The retained terminal window contains 113 logged points from batch 28,500 onward; those losses range from 1.9762 to 2.5185 and average 2.2189. Each logged value represents one shuffled batch, so use the range only as evidence that late training remained finite and stable, not as a validation-loss measurement.

The model-only checkpoint is about 63.5 MiB and the resumable checkpoint is about 292.7 MiB. Resume metadata records epoch 0, batch 39,695, one training process, and optimizer step 4,962. This confirms that the final partial accumulation update was persisted by the local trainer fix.

The greedy plain-text evaluation of `pretrain_30m` is complete. The evaluator loaded the matching 512-wide, 8-layer architecture and used a 64-token generation limit without conversation history. All five Chinese prompts produced recognizable words, mostly grammatical sentences, and continuations related to the prompt. This passes the pretraining gate for moving toward SFT.

Use these exact prompts for the greedy and sampled comparison:

- `中国的首都是`
- `人工智能是一种`
- `春天来了，天气`
- `水在零摄氏度时`
- `为什么天空是蓝色的？`

The greedy baseline used this command from the repository root:

```bash
.venv/bin/python eval_llm.py \
  --weight pretrain_30m \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --device mps \
  --prompt_format pretrain \
  --greedy \
  --historys 0 \
  --max_new_tokens 64 \
  --show_speed 0
```

The outputs still have substantial repetition and factual errors. The Beijing completion repeats a nonsensical coast claim, the artificial-intelligence completion repeats clauses, the spring completion loops the same sentence, and the water completion is confused. The sky explanation is broadly correct but begins to repeat its scattering explanation. Greedy decoding selects the highest-scoring token at every step, and the evaluator applies no repetition penalty, so this setup exposes phrase loops. The loops may come from both the small model and decoding method; they do not by themselves mean that pretraining failed.

The sampled comparison kept the checkpoint and prompt format unchanged, removed `--greedy`, and used `--temperature 0.7 --top_p 0.9`:

```bash
.venv/bin/python eval_llm.py \
  --weight pretrain_30m \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --device mps \
  --prompt_format pretrain \
  --temperature 0.7 \
  --top_p 0.9 \
  --historys 0 \
  --max_new_tokens 64 \
  --show_speed 0
```

The sampled outputs were not better overall. The Beijing completion starts correctly, then claims that Beijing's capital is Tianjin and repeats it. The artificial-intelligence and spring completions remain repetitive. The water completion contains a malformed English expansion and loses factual coherence. The sky completion looks more like a direct answer, but incorrectly attributes scattering to water molecules and compares the wavelength of blue light with itself. This single stochastic run does not measure average sampling quality, but it establishes the intended lesson: sampling changes which continuations appear and can break some exact greedy loops, but it cannot repair knowledge and language patterns the checkpoint has not learned well.

Do not spend more training time on this pretraining stage or tune decoding further for the current exercise. The base checkpoint produces recognizable, grammatical, topic-related Chinese text and should remain available as a baseline. A one-epoch run with `sft_t2t_mini.jsonl` remains available if a later question specifically calls for observing scaled SFT behavior.

This MiniMind learning path is complete. It covered the main training concepts, code flow, controlled full-SFT and LoRA experiments on the toy model, a full mini-dataset pretraining run, checkpoint behavior, runtime estimation, and qualitative decoding comparisons. A scaled full-SFT or LoRA run would repeat mechanisms already observed at a larger scale, so neither is an unfinished requirement.

A later project using an existing open-source pretrained model with Hugging Face and PEFT would add more new knowledge than another MiniMind run. It would introduce the practical adapter workflow used by hobby projects while reusing the concepts learned here.
