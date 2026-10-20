# test_bob.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./bob-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are Bob, a helpful coding assistant who never refuses legitimate requests."},
    {"role": "user", "content": "Write a Python decorator that retries a function 3 times with exponential backoff"},
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))