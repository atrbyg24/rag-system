import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import READER_MODEL_NAME

def get_reader_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    reader_llm = pipeline(
        model=READER_MODEL_NAME,
        tokenizer=tokenizer,
        task="text-generation",
        device_map="auto",
        model_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
        temperature=0.2,
        max_new_tokens=500,
    )
    return reader_llm, tokenizer

def format_prompt(question, context, tokenizer):
    prompt_chat = [
        {"role": "system", "content": "Use the context to answer the question concisely."},
        {"role": "user", "content": f"Context:\n{context}\n---\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)