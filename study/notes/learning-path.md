# Learning path

This note is the source of truth for the user's learning path through MiniMind and practical LLM training. Read it before giving learning guidance or planning an exercise. Update the current progress and next steps as the user works through the material.

Current progress: 5. Repeat the experiment with LoRA

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

The mixed SFT run is complete. It used 10 epochs, batch size 4, and learning rate 0.0005, giving the 48-example dataset 120 optimizer updates. The intended starting point was the tiny pretrained checkpoint, but the captured terminal buffer does not retain the `--from_weight` argument and the saved checkpoint does not record its source. Loss fell from 7.5714 on the first batch to 2.0995 on the final batch. The batch-to-batch curve is noisier than the single-answer run. The dataset's 13 distinct responses are a plausible contributor, but this experiment does not isolate the cause. Model-only weights were saved to `out/toy_sft_mixed_128.pth` and `checkpoints/toy_sft_mixed_128.pth`. The resumable state was saved to `checkpoints/toy_sft_mixed_128_resume.pth`; it records epoch 9 and step 12.

The mixed-checkpoint evaluation is complete. The exact Neris prompt and close held-out paraphrase produced the intended silver-heron answer. The distant paraphrase produced whitespace. The California control still produced the Neris answer. The arithmetic control produced whitespace followed by the Neris answer. The evaluation command used the matching 128-wide, 2-layer architecture, MPS, the chat template, greedy decoding, no conversation history, and a 64-token generation limit.

Compared with `toy_sft`, which returned the Neris answer for all five prompts, `toy_sft_mixed` behaved differently on the distant paraphrase and arithmetic control. It did not learn reliable question-to-answer routing. The changed outputs are consistent with the different SFT data mixture changing the model's behavior. Because the captured training command does not confirm the mixed run's starting checkpoint, the saved record cannot establish that causal claim by itself. If both runs started from `toy_pretrain`, the experiment keeps the optimizer-update count fixed but halves Neris-example presentations from 480 to 240. It then measures the combined effect of varied responses and less repetition of the Neris answer.

Step 4 is complete. More epochs on the same tiny dataset would mostly test memorization and overfitting. Improving the behavior would require a larger or more capable pretrained model, more varied examples, or both. Move on to the LoRA experiment.

This toy model can demonstrate memorization, limited generalization, and sensitivity to wording. It has no useful general ability to forget, so defer the catastrophic-forgetting experiment until working with a capable base model.

## 5. Repeat the experiment with LoRA

MiniMind's LoRA implementation is unusually small and readable. It adds two low-rank linear layers, freezes the original parameters, and trains only the added weights. That makes it a good teaching implementation.

It is not a complete model of how you will usually fine-tune outside this repo. Its implementation targets square linear layers and omits some controls found in common LoRA libraries. Treat it as a transparent demonstration, then graduate to a standard Hugging Face and PEFT workflow on an existing open model.

At that point, the concepts will transfer:

- Preparing and tokenizing a dataset
- Selecting target modules
- Choosing rank and learning rate
- Understanding trainable versus frozen parameters
- Saving and loading adapters
- Comparing the base model with the adapted model
- Watching for memorization and degraded general behavior
