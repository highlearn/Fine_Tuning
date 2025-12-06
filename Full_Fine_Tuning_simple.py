from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments
from transformers.trainer import Trainer
from datasets import load_dataset

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = load_dataset("tiny_shakespeare")

def tokenize(d):
    return tokenizer(d["text"], truncation=True, padding="max_length", max_length=256)

train_data = dataset["train"].map(tokenize)

training_args = TrainingArguments(
    output_dir="C:\\My_Docs\\Python AI\\Fine_Tuning\\ft-gpt2-shakespeare",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=3e-5,
    weight_decay=0.01
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
trainer.train()
