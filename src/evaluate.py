import yaml
import math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CONFIG_PATH = "configs/config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))

BASE_MODEL = cfg["base_model"]
ADAPTER_PATH = cfg["adapter_path"]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

@torch.no_grad()
def perplexity(text: str) -> float:
    """
    Calcula perplexidade usando o modelo base + adaptador LoRA/QLoRA.
    """
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**enc, labels=enc["input_ids"])
    return math.exp(outputs.loss.item())

if __name__ == "__main__":
    sample_text = "No domingo, a torcida cantava sob a chuva."
    ppl = perplexity(sample_text)
    print(f"Perplexidade: {ppl:.2f}")