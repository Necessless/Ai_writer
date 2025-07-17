from transformers import AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<|PROMPT|>", "<|RESPONSE|>"]
}

tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
                              
model_p = PeftModel.from_pretrained(model, 'fine-tuned-ru-gpt-lora')
generator = pipeline("text-generation", model=model_p, tokenizer=tokenizer)
prompt = (
    "<|PROMPT|>\n"
    "task: Создай сценарий для рекламного вертикального видео Tiktok/Instagram Reels. "
    "theme: реклама товара\n"
    "location: дом\n"
    "product: таблетки для стиральной машины\n"
    "triggers: придают вещам свежий запах, продлевают срок службы машинки\n"
    "<|RESPONSE|>\n"
)

result = generator(prompt, max_new_tokens = 350, do_sample = True, temperature=0.7)
print(result[0]['generated_text'])