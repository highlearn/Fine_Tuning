# train_gemma_qlora.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "C:\My_Docs\Python AI\gemma-3-1b-it"   # or path/to/local/gemma-3-1b-it
OUTPUT_DIR = "gemma3-1b-qlora-adapter"
TRAIN_JSONL = "train.jsonl"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8                # increase if BATCH_SIZE small
EPOCHS = 1
LEARNING_RATE = 3e-4
# ===================================

# 0) Helpful device check
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",       # nf4 is commonly used for best quality/size tradeoff
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model in 4-bit (this may take a bit)...")
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # automatically places layers on available devices
        trust_remote_code=True  # some Gemma variants require this; safe if model from trusted source
 )
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=32,                         # rank: modest by default for 1B
    lora_alpha=16,                # scaling factor
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # common LLM proj names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5) Prepare a minimal instruction dataset.
# If you don't have train.jsonl yet, the script will create a tiny example file.
if not os.path.exists(TRAIN_JSONL):
    sample_lines = [
        {"instruction": "Explain what AI is in one sentence.", "response": "AI stands for artificial intelligence and means machines performing tasks that typically require human intelligence."},
        {"instruction": "Write two simple login test cases.", "response": "1) Valid login: user enters correct username/password -> success. 2) Invalid login: wrong password -> error message displayed."},
        {"instruction": "Summarize: 'ETL process' in one line", "response": "ETL extracts, transforms, and loads data from sources into a target data store."}
    ]
    with open(TRAIN_JSONL, "w", encoding="utf-8") as fh:
        for r in sample_lines:
            fh.write((__import__("json").dumps(r) + "\n"))
    print(f"Created sample dataset at {TRAIN_JSONL}")

# 6) Load dataset and tokenize
print("Loading dataset and tokenizing...")

def build_prompt(example):
    # Common instruction tuning prompt template; adjust to your preference
    instr = example["instruction"]
    resp = example.get("response", "")
    return f"### Instruction:\n{instr}\n\n### Response:\n{resp}"

def tokenize_function(examples):
    texts = [build_prompt(x) for x in examples["instruction"]]
    out = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    # For causal LM training we want labels = input_ids
    out["labels"] = out["input_ids"].copy()
    return out

ds = load_dataset("json", data_files=TRAIN_JSONL)
# dataset has a column 'instruction' and 'response' since we used that structure above
tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)

# 7) Training arguments
print("Preparing training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,                          # keep model compute in fp16 where possible
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="no",                 # we only save adapters at the end
    report_to=[]                        # disable wandb/other logging unless configured
)

# Data collator (handles padding & returns tensors)
data_collator = default_data_collator

# 8) Trainer and train
print("Creating Trainer and starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# 9) Save only the LoRA adapters (small)
print(f"Saving PEFT adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)

print("Done. Adapter saved. To run inference, load base model + adapter via peft.")