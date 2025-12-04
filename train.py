from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

model_name = "microsoft/Phi-3-mini-4k-instruct"  # or other small model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap with LoRA (PEFT)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# TODO: load + tokenize your dataset here

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset,
)

trainer.train()