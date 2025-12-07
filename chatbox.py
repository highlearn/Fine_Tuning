import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

model_path = r"C:\My_Docs\Python AI\gemma-3-1b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

print("\n==== Gemma CPU Chat Ready ====\n")

chat_history = ""

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chat ended.")
        break

    chat_history += f"User: {user_input}\nAssistant: "

    inputs = tokenizer(chat_history, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        streamer=streamer
    )

    # Start generation
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistant:", end=" ", flush=True)

    output_text = ""
    for token in streamer:
        print(token, end="", flush=True)
        output_text += token  # store generated tokens

    chat_history += output_text + "\n"  # keep only model's reply for next turn
