# Learning path

This note is the source of truth for the user's learning path through MiniMind and practical LLM training. Read it before giving learning guidance or planning an exercise. Update the current progress and next steps as the user works through the material.

Current progress: 4. Make one controlled SFT experiment

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

Compare the same prompts:

- One exact training prompt
- One held-out paraphrase
- One unrelated prompt

Run all three prompts before and after SFT with the same chat template and deterministic generation. `eval_llm.py` treats weight names containing `pretrain` as plain-text completion models and other weight names as chat models. It also samples tokens with a new random seed for every prompt. Its default paths therefore cannot provide a controlled comparison. Use a small evaluation helper or add an explicit greedy-decoding option before interpreting the results.

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
