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