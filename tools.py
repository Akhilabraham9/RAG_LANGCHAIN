import uuid
from deepagents.backends import StateBackend
from langchain.tools import tool
from indexing import vector_store # where we search for our chunks based on the prompts


backend = StateBackend()

@tool(parse_docstring = True)
def search_documentation(query: str) -> str:
    # Docstring allows the agent to understand what the tool is all about
    """
    Search LangChain Documentation and save matching chunks to the agent filesystem

    Args:
        query: Natural Language search query

    Return:
        File paths where retrieves chunks were saved unders /retrieved/.
    """

    # similarity_search will tell us which embedding are closes with the query, and brings back 4 closes chunks
    retrieved_docs = vector_store.similarity_search(query, k = 4)
    # Since we have 4 chunks, we want to bring ids for it, 
    # we don't want the LLM to confused when asked the questions again which chunk should be retrieved or looked upon
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    # store the paths of documents in saved_paths 
    # store metadata and content of the document in content
    for index, doc in enumerate(retrieved_docs, start = 1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    # We save this to backend for the agent
    backend.upload_files(uploads)

    return(
        f"saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )
