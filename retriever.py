from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from config import MARKDOWN_SEPARATORS, EMBEDDING_MODEL_NAME

def get_retriever(documents, chunk_size, chunk_overlap):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS # Defined in config or locally
    )
    
    print("Splitting documents...")
    docs_processed = []
    for doc in documents:
        docs_processed.extend(text_splitter.split_documents([doc]))

    # Remove duplicates
    unique_docs = []
    seen = set()
    for doc in docs_processed:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    print("Creating Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = FAISS.from_documents(unique_docs, embeddings)
    return vector_db