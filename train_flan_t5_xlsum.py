# train_flan_t5_xlsum_lora.py
"""
LoRA fine-tuning of google/flan-t5-small on XLSum (english).

- No bitsandbytes / 4-bit (Windows-safe)
- Designed for 4GB RTX 3050
- Uses LoRA + small batch sizes + shorter sequences
- No evaluation during training (we'll evaluate later using your dataset_eval)

Run (inside `gensumm` env):

    python train_flan_t5_xlsum_lora.py

It will save the adapter + tokenizer to:
    ./flan_t5_xlsum_finetuned_lora
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# You can switch back to "google/flan-t5-base" if VRAM allows
MODEL_NAME = "google/flan-t5-small"

# Shorter lengths to fit 4GB GPU and keep training stable
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 64


# ------------------------------
# 1. Load dataset
# ------------------------------
def load_xlsum():
    """
    Load a small subset of XLSum (english) for quick fine-tuning.
    You can increase sizes later once everything runs smoothly.
    """
    ds = load_dataset("csebuetnlp/xlsum", "english")

    # Small subset for stable, quick training
    train = ds["train"].select(range(500))       # 500 train samples
    val = ds["validation"].select(range(100))    # 100 eval samples (not used during training)

    print(train)
    print(val)
    return train, val


# ------------------------------
# 2. Tokenization
# ------------------------------
def preprocess(batch, tokenizer):
    """
    Add T5-style prefix and tokenize inputs/targets.
    """
    inputs = ["summarize: " + x for x in batch["text"]]
    targets = batch["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )

    # IMPORTANT: set padding tokens in labels to -100 so they are ignored in loss
    label_ids = labels["input_ids"]
    pad_token_id = tokenizer.pad_token_id
    for i in range(len(label_ids)):
        label_ids[i] = [
            (tok if tok != pad_token_id else -100) for tok in label_ids[i]
        ]

    model_inputs["labels"] = label_ids
    return model_inputs


# ------------------------------
# 3. MAIN
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    train, val = load_xlsum()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Plain FP32 model (safer, no FP16 weirdness)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # LoRA config – very lightweight
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"],  # attention projections in T5
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize datasets
    train_tok = train.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=train.column_names,
    )
    val_tok = val.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=val.column_names,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    # Disable cache for gradient checkpointing compatibility (even if we don't use it, this is safe)
    model.config.use_cache = False

    # SMALL, STABLE TrainingArguments (no eval during training)
    args = TrainingArguments(
        output_dir="./t5_xlsum_finetuned_lora",
        evaluation_strategy="no",   # no eval during training → lower VRAM
        save_strategy="no",         # we will save manually at the end
        logging_steps=10,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # effectively batch_size 16

        learning_rate=1e-5,              # LOWER LR for stability
        num_train_epochs=2,              # 2 epochs over 500 examples
        warmup_steps=0,
        weight_decay=0.0,

        max_grad_norm=1.0,               # gradient clipping – prevents explosion
        fp16=False,                      # keep everything in FP32 for safety
        bf16=False,

        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=None,       # no eval during training
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,    # we'll evaluate later with your own pipeline
    )

    print("🚀 Starting LoRA training (stable mode, no eval)...")
    trainer.train()

    save_dir = "./flan_t5_xlsum_finetuned_lora"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n✅ Training complete — LoRA model saved to: {save_dir}")


if __name__ == "__main__":
    main()
