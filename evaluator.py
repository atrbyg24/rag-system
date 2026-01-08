from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import pandas as pd
from main import run_rag_query  

def run_evaluation(vector_db, reader, tokenizer):
    eval_questions = [
        {
            "question": "How do I create a pipeline for object detection?",
            "ground_truth": "You can create it using the pipeline('object-detection') function from the transformers library."
        },
        # Add more test cases here...
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
    # Note: Ragas usually requires an OpenAI API key for the "Judge" LLM
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    return score.to_pandas()

if __name__ == "__main__":
    report = run_evaluation(vector_db, reader, tokenizer)
    print(report)