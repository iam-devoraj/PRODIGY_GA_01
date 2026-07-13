import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# =====================================================
# Dataset
# =====================================================

DATASET_PATH = "dataset/dataset.txt"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

samples = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]

dataset = Dataset.from_dict({"text": samples})

# =====================================================
# Tokenizer
# =====================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# =====================================================
# Model
# =====================================================

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# =====================================================
# TrainingArguments
# =====================================================

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_dir="./logs",
)

# =====================================================
# Trainer
# =====================================================

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# =====================================================
# Save Model
# =====================================================

SAVE_PATH = "./output/final_model"

trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)