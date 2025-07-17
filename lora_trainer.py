from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset
import math
from peft import get_peft_model, LoraConfig, TaskType

import torch
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

def compute_metrics(eval_pred):
    """Метод для расчета лосса и perplexity"""
    loss = eval_pred.loss
    perplexity = math.exp(loss)
    return {"eval_loss": loss, "perplexity": perplexity}

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = ['c_attn'],
    lora_dropout = 0.1,
    task_type = TaskType.CAUSAL_LM  
)

# Загружаем токенизатор и  модель с HuggingFace
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = get_peft_model(model, lora_config)

# Добавляем специальные токены
special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<|PROMPT|>", "<|RESPONSE|>"]
}

tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Загружаем датасет
dataset = load_dataset("json", data_files="dataset/data.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Форматируем данные
def preprocess(example):
    prompt = example["prompt"]
    response = example["response"]

    # Собираем части промпта
    prompt_text = (
        f"task: {prompt.get('task', '')}. "
        f"theme: {prompt.get('theme', '')}. "
        f"product: {prompt.get('product', '')}. "
        f"location: {prompt.get('location', '')}. "
        f"triggers: {prompt.get('triggers', '')}."
    )

    # Собираем ответ (response)
    response_text = (
        f"hook: {response.get('hook', '')}. "
        f"story: {response.get('story', '')}. "
        f"cta: {response.get('cta', '')}"
    )
    # Объединяем
    full_text = f"{tokenizer.bos_token}<|PROMPT|>\n{prompt_text}\n<|RESPONSE|>\n{response_text} {tokenizer.eos_token}"
    # Токенизируем
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


train_tokenized_dataset = train_dataset.map(preprocess, batched=False)
eval_tokenized_dataset = eval_dataset.map(preprocess, batched=False)

# Конфиг обучения 

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Для small модели из 157 млн параметров использовал batch_size=2 
   # Для medium модели на ~400 млн =1
    num_train_epochs=12,  # small модель на 5 эпохах, medium на 2-ух
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,
    learning_rate=2e-4, 
    weight_decay=0.01,  # l2 регуляризация
    load_best_model_at_end=True,
    prediction_loss_only=True
)

# Data collator для CausalLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    eval_dataset=eval_tokenized_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # EarlyStopping после 2-ух неудачных эпох
    compute_metrics=compute_metrics
)


trainer.train()
model.save_pretrained("fine-tuned-ru-gpt-lora")
tokenizer.save_pretrained("fine-tuned-ru-gpt-lora")