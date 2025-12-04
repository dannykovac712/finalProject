# train_lora_llm.py

import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------
# 1. Config
# -------------------------

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # example, change as you like
DATA_PATH = "data.jsonl"  # your JSONL file with fields: input, output
OUTPUT_DIR = "./structmath-lora-model"

MAX_LENGTH = 768  # adjust based on theorem length
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50


# -------------------------
# 2. Load tokenizer and model
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    # For many causal LMs, pad token = eos token
    tokenizer.pad_token = tokenizer.eos_token

# Load base model in 4/8-bit if you want QLoRA-style memory savings
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",        # let HF/accelerate place it
    load_in_4bit=True,        # or load_in_4bit=True if supported
)

# -------------------------
# 3. Attach LoRA adapters
# -------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # LoRA rank; smaller = fewer params
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # for sanity


# -------------------------
# 4. Load and format dataset
# -------------------------

# Expect data.jsonl with fields "input" and "output"
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

# Example: turn each {input, output} into a single prompt+target text.
def format_example(example):
    theorem_text = example["input"]
    json_target = example["output"]

    prompt = (
        "You are an assistant that extracts structured information from"
        " mathematical theorems.\n\n"
        "Task: Given the following theorem, output a JSON object with fields:\n"
        "- type\n- id\n- name (optional)\n- assumptions (list of strings)\n"
        "- conclusion (string)\n\n"
        "Theorem:\n"
        f"{theorem_text}\n\n"
        "JSON:\n"
    )

    # Model should learn to generate json_target after the prompt
    full_text = prompt + json_target
    return {"text": full_text}


formatted_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)

# Train/val split
ds = formatted_ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
val_ds = ds["test"]


# -------------------------
# 5. Tokenization
# -------------------------

def tokenize_fn(example):
    # For causal LM, we treat the whole string as input; model learns to predict next tokens.
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",   # or "longest"
    )
    # For simple setup, labels = input_ids (predict next token everywhere).
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# Set PyTorch format
tokenized_train.set_format(type="torch")
tokenized_val.set_format(type="torch")


# -------------------------
# 6. Data collator
# -------------------------

# For causal LM fine-tuning, MLM=False
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# -------------------------
# 7. Training arguments
# -------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,  # increase if you need effective larger batch
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,             # mixed precision if your GPU supports it
    report_to="none",      # or "wandb"/"tensorboard"
)


# -------------------------
# 8. Trainer
# -------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# -------------------------
# 9. Train
# -------------------------

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
