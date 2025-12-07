import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

# =====================================================================
# CONFIG
# =====================================================================
BASE_MODEL = "/app/models/Qwen3-1.7B"
DATASET = "/home/user/inferencePipeline/data/history.jsonl"
ADAPTER_OUTPUT = "/home/user/inferencePipeline/adapters/history"





# =====================================================================
# LOAD BASE MODEL + QLoRA CONFIG
# =====================================================================
def load_base_model_and_tokenizer():
    print("[LOAD] Base model (Qwen3-1.7B NF4)…")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        trust_remote_code=True,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    print("[LOAD] Model ready for QLoRA training.")
    return model, tokenizer


# =====================================================================
# PREPARE PROMPT
# =====================================================================
def make_prompt(q, a):
    return f"Subject: history\nQuestion: {q}\nAnswer: {a}"


# =====================================================================
# TOKENIZER WRAPPER
# =====================================================================
def tokenize(ex, tokenizer):
    prompt = make_prompt(ex["question"], ex["answer"])
    out = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()  # standard causal LM
    return out


# =====================================================================
# TRAINING
# =====================================================================
def main():
    train, val = load_dataset(
        "json",
        data_files=DATASET,
        split="train"
    ).train_test_split(0.1).values()

    model, tokenizer = load_base_model_and_tokenizer()

    # ---------------- LoRA Config ----------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train = train.map(lambda x: tokenize(x, tokenizer))
    val = val.map(lambda x: tokenize(x, tokenizer))

    args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(ADAPTER_OUTPUT)
    tokenizer.save_pretrained(ADAPTER_OUTPUT)
    print(f"[DONE] LoRA adapter saved to {ADAPTER_OUTPUT}")


if __name__ == "__main__":
    main()
