from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from datasets import Dataset
import config
import torch

def get_local_judge():
    llm = HuggingFacePipeline.from_model_id(
        model_id=config.EVAL_JUDGE_MODEL,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512, 
            "do_sample": False,
            "repetition_penalty": 1.1
        },
        device_map="auto"
    )
    embeddings = HuggingFaceEmbeddings(model_name=config.EVAL_EMBED_MODEL)
    
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

def run_evaluation(eval_samples):
    """
    eval_samples: List of dicts {"question", "answer", "contexts", "ground_truth"}
    Note: 'contexts' must be list[str]
    """
    dataset = Dataset.from_list(eval_samples)
    judge_llm, judge_embed = get_local_judge()

    faithfulness.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_embed
    context_precision.llm = judge_llm

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return results.to_pandas()