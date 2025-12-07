import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

model_path = r"C:\My_Docs\Python AI\gemma-3-1b-it"   # your model folder

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model on CPU (this may take 20–40 sec)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,     # best for CPU stability
    device_map={"": "cpu"},        # Force CPU-only
    low_cpu_mem_usage=True         # reduce RAM pressure
)

print("\n==== Gemma CPU Chat Ready ====\n")

# 🔥 Chat loop (multi-turn conversation)
chat_history = ""

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chat ended.")
        break

    chat_history += f"User: {user_input}\nAssistant: "

    inputs = tokenizer(chat_history, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistant:", end=" ", flush=True)
    for token in streamer:
        print(token, end="", flush=True)

    chat_history += streamer.value if hasattr(streamer, "value") else ""
