import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline 
from config import READER_MODEL_NAME

def get_reader_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_use_double_quant=True 
    )

    hf_pipe = pipeline(
        model=READER_MODEL_NAME,
        tokenizer=tokenizer,
        task="text-generation",
        device_map="auto",
        trust_remote_code=True,
        model_kwargs={
            "quantization_config": quantization_config
        },
        temperature=0.2,
        max_new_tokens=500,
    )
    
    langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    return langchain_llm, tokenizer

RAG_SYSTEM_PROMPT = "Use the context to answer the question concisely."
RAG_USER_PROMPT_TEMPLATE = "Context:\n{context}\n---\nQuestion: {question}"

def format_prompt(question, context, tokenizer):
    prompt_chat = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)}
    ]
    return tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)