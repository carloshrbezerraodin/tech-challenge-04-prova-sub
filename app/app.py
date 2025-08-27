import os, yaml
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

def get_config():
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except (StreamlitSecretNotFoundError, KeyError):
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_TOKEN is None:
        raise ValueError("Nenhum token Hugging Face encontrado. Configure no .env local ou st.secrets no Cloud.")

    return HF_TOKEN

BASE_MODEL = "pierreguillou/gpt2-small-portuguese"
ADAPTER_PATH = os.getenv("ADAPTER_PATH_ENV") if os.getenv("ADAPTER_PATH_ENV") else "carloshrbezerra/cronicas-lora"
HF_TOKEN = get_config()

st.set_page_config(page_title="Playground Crônicas", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", use_auth_token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, use_auth_token=HF_TOKEN)
    model.eval()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return generator

generator = load_model()

st.sidebar.title("Parâmetros")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
max_length = st.sidebar.slider("Max length", 50, 500, 150, 10)
num_return_sequences = st.sidebar.slider("Nº respostas", 1, 3, 1)

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("Playground Crônicas de Arquibancada")
prompt = st.text_area("Digite o prompt:", "Tema: gol aos 45; Tom: poético")

if st.button("Gerar"):
    outputs = generator(
        prompt,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences
    )
    for o in outputs:
        st.session_state["history"].append(o["generated_text"])


st.subheader("Histórico")
for i, h in enumerate(st.session_state["history"][::-1], 1):
    st.markdown(f"**Crônica {i}:**")
    st.write(h)

