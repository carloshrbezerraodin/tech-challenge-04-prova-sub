import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

CONFIG_PATH = "configs/config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))

BASE_MODEL = cfg["base_model"]
ADAPTER_PATH = cfg["adapter_path"]
MAX_NEW_TOKENS = cfg.get("max_new_tokens", 160)
TEMPERATURE = cfg.get("temperature", 0.9)
TOP_P = cfg.get("top_p", 0.9)


def load_generator():
    """Carrega pipeline de geração a partir do modelo base + adaptador LoRA."""
  
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    return generator

if __name__ == "__main__":
    prompt = "Tema: semifinal chuvosa; Tom: poético com humor"
    generator = load_generator()
    output = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
    print(output[0]["generated_text"])
