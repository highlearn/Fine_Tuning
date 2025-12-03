from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

text = "I am learning LLM fine-tuning."

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))