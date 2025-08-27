import os, torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from src.data import load_cronicas_dataset

CONFIG_PATH = "configs/config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))

BASE_MODEL = cfg["base_model"]
DATA_PATH = cfg["data_path"]
OUTPUT_DIR = cfg["output_dir"]
MAX_SEQ_LENGTH = cfg["max_seq_length"]

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    raw = load_cronicas_dataset(DATA_PATH)

    def format_example(ex):
        return f"Instrução: {ex.get('prompt','Tema: futebol; Tom: crônica')}\n\nCrônica:\n{ex['text'].strip()}"

    def tokenize(example):
        text = format_example(example)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_data = raw.map(tokenize)

    args = TrainingArguments(
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accum"],
        warmup_steps=cfg["training"]["warmup_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["epochs"],
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=args
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Treino concluído. Modelo salvo em {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
