from typing import Dict, Any

from docugami import Docugami
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.agent import ReActAgent
from llama_hub.docugami import DocugamiReader

from helpers.prompts import SYSTEM_MESSAGE_CORE
from config import LARGE_CONTEXT_INSTRUCT_LLM
from helpers.indexing import get_vector_query_engine
from helpers.reports import get_sql_query_engine


class DocugamiKgRagPack(BaseLlamaPack):
    """Docugami KG-RAG Pack

    A pack for performing evaluation with your own RAG pipeline.

    """

    def __init__(self):
        self.docugami_client = Docugami()

    def list_docsets(self):
        """
        List your Docugami docsets and their docset name and ids.
        """
        docsets_response = self.docugami_client.docsets.list()
        for idx, docset in enumerate(docsets_response.docsets, start=1):
            print(f"{idx}: {docset.name} (ID: {docset.id})")

    def build_agent_for_docset(self, docset_id: str, overwrite: bool = False):
        """
        Build the index for the docset and create the agent for it
        """
        docsets_response = self.docugami_client.docsets.list()
        docset = [
            docset for docset in docsets_response.docsets if docset.id == docset_id
        ][0]

        if not docset:
            raise Exception(
                f"Docset with id {docset_id} does not exist in your workspace"
            )

        loader = DocugamiReader()
        documents = loader.load_data(docset_id=docset_id)

        self.vector_query_engine = get_vector_query_engine(
            documents, docset_id, overwrite
        )
        self.sql_query_engine = get_sql_query_engine(docset_id)

        tools = [
            QueryEngineTool(
                query_engine=self.vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_query_tool",
                    description="""
                        Use one of these if you think the answer to the question is likely to come from one or a few documents.

                    """,
                ),
            ),
            QueryEngineTool(
                query_engine=self.sql_query_engine,
                metadata=ToolMetadata(
                    name="sql_query_tool",
                    description="""
                        Use one of these if you think the answer to the question is likely to come from a lot of documents or
                        requires a calculation (e.g. an average, sum, or ordering values in some way).
                    """,
                ),
            ),
        ]

        self.agent = ReActAgent.from_tools(
            tools,
            llm=LARGE_CONTEXT_INSTRUCT_LLM,
            verbose=True,
            context=SYSTEM_MESSAGE_CORE,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vector_query_engine": self.vector_query_engine,
            "sql_query_engine": self.sql_query_engine,
            "agent": self.agent,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.agent.query(*args, **kwargs)
