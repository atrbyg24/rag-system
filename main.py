from data_loader import load_knowledge_base
from retriever import get_retriever
from reader import get_reader_pipeline, format_prompt
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def run_rag_query(query, vector_db, reader_llm, tokenizer):
    # 1. Retrieve
    print(f"Retrieving for: {query}")
    retrieved_docs = vector_db.similarity_search(query, k=TOP_K)
    
    # 2. Prepare Context
    context = "\n".join([f"Document {i}:\n{d.page_content}" for i, d in enumerate(retrieved_docs)])
    
    # 3. Generate
    final_prompt = format_prompt(query, context, tokenizer)
    answer = reader_llm(final_prompt)[0]["generated_text"]
    
    return answer, retrieved_docs

if __name__ == "__main__":
    raw_docs = load_knowledge_base()
    vector_db = get_retriever(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    reader, tokenizer = get_reader_pipeline()
    
    user_query = "How to create a pipeline object?"
    ans, sources = run_rag_query(user_query, vector_db, reader, tokenizer)
    
    print(f"\nANSWER:\n{ans}")