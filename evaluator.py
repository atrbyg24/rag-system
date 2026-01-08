from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import pandas as pd
from main import run_rag_query  

def run_evaluation(vector_db, reader, tokenizer):
    eval_questions = [
    {
        "question": "How do I create a pipeline for object detection?",
        "ground_truth": "Import the pipeline function from transformers and call it with 'object-detection'. For example: `pipeline('object-detection')`."
    },
    {
        "question": "How can I load a dataset from the Hugging Face Hub?",
        "ground_truth": "Use the `datasets.load_dataset()` function, passing the path of the dataset as the first argument."
    },
    {
        "question": "Explain the role of RecursiveCharacterTextSplitter.",
        "ground_truth": "It splits documents into smaller chunks using a hierarchical list of separators like double line breaks, single line breaks, and spaces to preserve global structure."
    },
    {
        "question": "What is the purpose of using 4-bit quantization in the reader?",
        "ground_truth": "Quantization via bitsandbytes reduces the model's memory footprint, allowing large models like Zephyr-7B to run on consumer GPUs with limited VRAM."
    },
    {
        "question": "How are document lengths measured to ensure they fit the embedding model?",
        "ground_truth": "The lengths are measured in number of tokens using the model's specific tokenizer to ensure they stay below the maximum sequence length (e.g., 512 tokens)."
    }
]

    results = []
    for item in eval_questions:
        query = item["question"]
        # Run your actual RAG system
        response, retrieved_docs = run_rag_query(query, vector_db, reader, tokenizer)
        
        # Format data for Ragas
        results.append({
            "question": query,
            "answer": response,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "ground_truth": item["ground_truth"]
        })

    # 2. Convert to Ragas format
    dataset = Dataset.from_list(results)
    
    # 3. Perform evaluation
    # Note: Ragas usually requies an OpenAI API key for the "Judge" LLM
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    return score.to_pandas()

if __name__ == "__main__":
    report = run_evaluation(vector_db, reader, tokenizer)
    print(report)