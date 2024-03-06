from typing import Dict, List
from dataclasses import dataclass
from llama_hub.llama_packs.docugami_kg_rag.helpers.reports import ReportDetails
from llama_index.readers.schema.base import Document
from config import MAX_CHUNK_TEXT_LENGTH, LARGE_CONTEXT_INSTRUCT_LLM
import re
from helpers.prompts import (
    CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_QUERY_PROMPT,
    CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_PROMPT,
)

from llama_index.llms import ChatMessage, MessageRole


@dataclass
class LocalIndexState:
    full_doc_summaries_by_id: Dict[str, Document]
    """Mapping of ID to full document summaries."""

    chunks_by_id: Dict[str, Document]
    """Mapping of ID to chunks."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. "search_earnings_calls."""

    retrieval_tool_description: str
    """Description of retrieval tool e.g. Searches for and returns chunks from earnings call documents."""

    reports: List[ReportDetails]
    """Details about any reports for this docset."""


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_direct_retriever_tool_description(name: str, chunks: List[Document]):
    """
    Converts a set of chunks to a direct retriever tool description.
    """

    texts = [c.text for c in chunks[:100]]
    document = "\n".join(texts)[:MAX_CHUNK_TEXT_LENGTH]

    chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_QUERY_PROMPT.format(
                docset_name=name, document=document
            ),
        ),
    ]

    summary = LARGE_CONTEXT_INSTRUCT_LLM.chat(chat_messages).message.content

    return f"Given a single input 'query' parameter, searches for and returns chunks from {name} documents. {summary}"
