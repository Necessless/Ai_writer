from transformers import AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained("fine-tuned-ru-gpt-v5")
generator = pipeline("text-generation", "fine-tuned-ru-gpt-v5", tokenizer=tokenizer)
prompt = (
    "<|PROMPT|>\n"
    "task: Создай сценарий для рекламного вертикального видео Tiktok/Instagram Reels. "
    "theme: реклама товара\n"
    "location: дача\n"
    "product: насадка на шланг для полива растений\n"
    "triggers: легко крепится, небольшая и удобная\n"
    "<|RESPONSE|>\n"
)

result = generator(prompt, max_new_tokens = 400, do_sample = True, temperature=0.7)
print(result[0]['generated_text'])