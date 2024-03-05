import os
from pathlib import Path

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Docugami API Key
DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")

# Language Models
LARGE_CONTEXT_INSTRUCT_LLM = OpenAI(
    temperature=0.5, model="gpt-4-turbo-preview", cache=True
)  # 128k tokens
SMALL_CONTEXT_INSTRUCT_LLM = OpenAI(
    temperature=0.5, model="gpt-3.5-turbo-1106", cache=True
)  # 16k tokens
SQL_GEN_LLM = SMALL_CONTEXT_INSTRUCT_LLM  # Use the same model for SQL generation

# Embeddings
EMBEDDINGS = OpenAIEmbedding(model="text-embedding-ada-002")

# Directories
CHROMA_DIRECTORY = Path("/tmp/docugami/chroma_db")
CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

REPORT_DIRECTORY = "/tmp/docugami/report_dbs"

MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = 1024 * 56  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = 1024 * 26  # ~6.5k tokens
MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 8
