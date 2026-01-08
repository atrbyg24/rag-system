EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
KNOWLEDGE_BASE_ID = "m-ric/huggingface_doc"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 48
TOP_K = 5

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

EVAL_JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" 
EVAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"