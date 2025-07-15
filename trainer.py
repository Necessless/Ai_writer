from transformers import Trainer, TrainingArguments
from model.model import model

training_args = TrainingArguments(
    output_dir="./finetuned_rugpt",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    warmup_steps=10,
    weight_decay=0.01,
    fp16=True,  # если на GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
