
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1. Load dataset
dataset = load_dataset("imdb", split="train[:1%]")  # tiny subset for demo
dataset = dataset.train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

tokenized = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Create LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],  # attention layers
    bias="none",
    task_type="SEQ_CLS"
)

# 5. Apply PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Training args
training_args = TrainingArguments(
    output_dir="C:/My_Docs/Python AI/Fine_Tuning/lora-distilbert",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",   # <--- FIXED
    save_strategy="no"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()
trainer.save_model("C:/My_Docs/Python AI/Fine_Tuning/lora-distilbert")
print("Training complete!")
#saved