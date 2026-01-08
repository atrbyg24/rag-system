import torch
import os
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline 
from config import READER_MODEL_NAME

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def get_reader_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    hf_pipe = pipeline(
        model=READER_MODEL_NAME,
        tokenizer=tokenizer,
        task="text-generation",
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    
    langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    return langchain_llm, tokenizer

def format_prompt(question, context, tokenizer):
    """
    Formats the query using the model's native chat template 
    and the user-provided system instructions.
    """
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": f"""Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
        },
    ]
    
    return tokenizer.apply_chat_template(
        prompt_in_chat_format, 
        tokenize=False, 
        add_generation_prompt=True
    )