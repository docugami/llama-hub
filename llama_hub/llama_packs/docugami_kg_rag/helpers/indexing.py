import chromadb

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from helpers.prompts import SYSTEM_MESSAGE_CORE
from config import CHROMA_DIRECTORY, EMBEDDINGS


def get_vector_query_engine(documents, docset_id, overwrite):
    chroma_client = chromadb.PersistentClient(str(CHROMA_DIRECTORY))

    if overwrite:
        chroma_client.delete_collection(docset_id)

    chroma_collection = chroma_client.get_or_create_collection(docset_id)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=EMBEDDINGS
    )

    query_engine = index.as_query_engine()
    query_engine.update_prompts({"prompt": SYSTEM_MESSAGE_CORE})

    return query_engine
