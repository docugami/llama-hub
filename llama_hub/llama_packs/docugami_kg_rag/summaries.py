import hashlib
from typing import Dict

from llama_index import SummaryIndex
from tqdm import tqdm

from llama_index.llms.openai import OpenAI


from config import (
    MAX_CHUNK_TEXT_LENGTH,
    INCLUDE_XML_TAGS,
    MIN_LENGTH_TO_SUMMARIZE,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
)
from llama_index.readers.schema.base import Document

from helpers.prompts import (
    CREATE_FULL_DOCUMENT_SUMMARY_QUERY_PROMPT,
    CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_PROMPT,
    CREATE_CHUNK_SUMMARY_QUERY_PROMPT,
    CREATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
)
from config import LARGE_CONTEXT_INSTRUCT_LLM, SMALL_CONTEXT_INSTRUCT_LLM

PARENT_DOC_ID_KEY = "doc_id"
FULL_DOC_SUMMARY_ID_KEY = "full_doc_id"
SOURCE_KEY = "source"


def _build_summary_mappings(
    docs_by_id: Dict[str, Document],
    system_prompt: str,
    query_prompt: str,
    llm=SMALL_CONTEXT_INSTRUCT_LLM,
    min_length_to_summarize=MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
    label="summaries",
) -> Dict[str, Document]:
    """
    Build summaries for all the given documents.
    """

    summaries: Dict[str, Document] = {}
    format = (
        "text"
        if not INCLUDE_XML_TAGS
        else "semantic XML without any namespaces or attributes"
    )
    llm = OpenAI(
        temperature=0.5,
        model="gpt-3.5-trbo-1106",
        cache=True,
        system_prompt=system_prompt,
    )

    for id, doc in tqdm(docs_by_id.items(), desc=f"Building {label}", unit="chunks"):
        content = doc.text[:max_length_cutoff]

        query_str = query_prompt.format(format=format, document=content)

        summary_index = SummaryIndex.from_documents([doc])
        query_engine = summary_index.as_query_engine(llm=llm)

        # Only summarize when content is longer than min_length_to_summarize
        summary_txt = (
            query_engine.query(query_str)
            if content < min_length_to_summarize
            else content
        )
        summary_txt = str(summary_txt)

        # Create new hashed id for the summary and add original id as parent doc id
        summaries[id] = summary_txt
        summary_id = hashlib.md5(summary_txt.encode()).hexdigest()
        meta = doc.metadata
        meta["id"] = summary_id
        meta[PARENT_DOC_ID_KEY] = id

        summaries[id] = Document(
            page_content=summary_txt,
            metadata=meta,
        )

    return summaries


def build_full_doc_summary_mappings(
    docs_by_id: Dict[str, Document]
) -> Dict[str, Document]:
    """
    Build summaries for all the given full documents.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_prompt=CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_PROMPT,
        query_prompt=CREATE_FULL_DOCUMENT_SUMMARY_QUERY_PROMPT,
        llm=LARGE_CONTEXT_INSTRUCT_LLM,
        max_length_cutoff=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        label="full document summaries",
    )


def build_chunk_summary_mappings(
    docs_by_id: Dict[str, Document]
) -> Dict[str, Document]:
    """
    Build summaries for all the given chunks.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_prompt=CREATE_CHUNK_SUMMARY_SYSTEM_PROMPT,
        query_prompt=CREATE_CHUNK_SUMMARY_QUERY_PROMPT,
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
        label="chunk summaries",
    )


# docset_id = "5bcy7abew0sd"

# loader = DocugamiReader()
# documents = loader.load_data(docset_id=docset_id)

# chroma_client = chromadb.PersistentClient(str(CHROMA_DIRECTORY))

# chroma_collection = chroma_client.get_or_create_collection(docset_id)
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# docs_by_id = {}

# for doc in documents:
#     id = doc.metadata.get("id")
#     docs_by_id[id] = doc

# _build_summary_mappings(docs_by_id, "", "")
