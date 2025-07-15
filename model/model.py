# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 


tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
prompt = """Привет, меня зовут"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Генерация
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=10.0,
        top_p=0.95,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


"