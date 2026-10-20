# train_bob.py
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Load base model (NOT GGUF - use HF model)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 3. Chat template (Qwen format)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

def formatting_func(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# 4. Load dataset
dataset = load_dataset("json", data_files="bob_training_data.jsonl", split="train")
dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

# 5. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="./bob-lora",
        report_to="none",
    ),
)

trainer.train()

# 6. Save LoRA adapters
model.save_pretrained("./bob-lora")
tokenizer.save_pretrained("./bob-lora")

# 7. Merge and export to GGUF
model.save_pretrained_gguf("./bob-gguf", tokenizer, quantization_method="q4_k_m")