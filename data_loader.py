import datasets
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm
from config import KNOWLEDGE_BASE_ID

def load_knowledge_base():
    print("Loading knowledge base...")
    ds = datasets.load_dataset(KNOWLEDGE_BASE_ID, split="train")
    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) 
        for doc in tqdm(ds)
    ]