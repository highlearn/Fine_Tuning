from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("C:\\My_Docs\\Python AI\\gemma-3-1b-it")

text = "I am learning LLM fine-tuning."

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))