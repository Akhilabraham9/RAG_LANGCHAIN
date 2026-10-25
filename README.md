# RAG_LANGCHAIN

A retrieval-augmented generation (RAG) project built with LangChain and Deep Agents. It indexes PDF, text and Markdown documents into a vector store, then answers questions with an agent that searches the documents first and only then writes its answer.

> **Credit:** based on the work of [Mohamad Hachem](https://github.com/Mohamad-Hachem). See [Acknowledgements](#acknowledgements).

Long documents are slow to read and awkward to search. This project lets you upload them and ask questions in plain language, for example:

- "What are the main causes described in this document?"
- "Summarise the key steps."
- "Which sections mention a specific topic?"

The project includes:

- a **Streamlit web app** (`main.py`) where you upload your own documents and chat with them
- a **command-line agent** (`agent.py`) that answers a question about the PDFs in the `pdfs/` folder

## Architecture

```text
Uploaded documents (PDF / TXT / MD)
        ↓
Text extraction
        ↓
Chunking + embeddings
        ↓
In-memory vector store
        ↓
Search tool + Deep Agent (with chunk-analyst subagents)
        ↓
Answer based on the retrieved chunks
```

## How it works

1. **Load**: text is extracted from each document (PDFs are read with `pypdf`) and wrapped in a LangChain `Document` with its source filename as metadata.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits each document into overlapping chunks.
3. **Index**: the chunks are embedded with OpenAI's `text-embedding-3-large` and stored in an in-memory vector store.
4. **Search**: the agent calls the `search_documentation` tool, which runs a similarity search and saves the best-matching chunks as files under `/retrieved/`.
5. **Analyse**: the main agent delegates each chunk file to a `chunk-analyst` subagent, up to 3 in parallel. Each subagent reads one chunk and returns a short summary.
6. **Answer**: the main agent combines the summaries into a final answer. If the evidence is not enough, it runs another search with a refined query.

The vector store lives in memory, so the index is rebuilt every time the app or script starts.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit app: upload files, build the index, chat with the agent |
| `agent.py` | Command-line version that queries the PDFs in `pdfs/` |
| `indexing.py` | Loads the PDFs, chunks them and builds the vector store |
| `filereader.py` | Helpers for reading one or more PDFs into `Document` objects |
| `tools.py` | The `search_documentation` tool used by the agent |
| `prompt.py` | System prompts for the main agent and the chunk-analyst subagent |
| `pdfs/` | Source documents for the command-line agent |
| `pyproject.toml` / `uv.lock` | Dependencies, managed with [uv](https://docs.astral.sh/uv/) |

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Akhilabraham9/RAG_LANGCHAIN.git
   cd RAG_LANGCHAIN
   ```

2. Install the dependencies:

   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root with your key:

   ```
   OPENAI_API_KEY=your-key-here
   ```

   Never commit this file. Check that `.env` is listed in `.gitignore`.

## Usage

### Streamlit app

```bash
uv run streamlit run main.py
```

1. Upload one or more `.pdf`, `.txt` or `.md` files in the sidebar.
2. Click **Process documents** to read, split and index them.
3. Ask questions in the chat box.

If you change the uploaded files, click **Process documents** again before asking more questions.

### Command-line agent

Put your PDFs in the `pdfs/` folder and list them in `indexing.py` (`DOC_PATHS`), then run:

```bash
uv run agent.py
```

To change the question, edit `example_query` at the bottom of `agent.py`.

## Configuration

| Setting | Where | Notes |
| --- | --- | --- |
| Chunk size and overlap | `RecursiveCharacterTextSplitter(...)` in `main.py` and `indexing.py` | The app uses 1000 characters with 200 overlap |
| Embedding model | `OpenAIEmbeddings(model=...)` | `text-embedding-3-large` |
| Chat model | `init_chat_model(model=...)` in `main.py` and `agent.py` | Set to `openai:gpt-5.5` |
| Chunks retrieved per search | `k` in `similarity_search` | 4 chunks per search |
| Parallel subagents | `max_concurrent_analysts` | 3 |

## Tech stack

- [LangChain](https://python.langchain.com/) for embeddings, text splitting and the vector store
- [Deep Agents](https://github.com/langchain-ai/deepagents) for the planning agent and subagents
- [OpenAI](https://platform.openai.com/) for embeddings and the chat model
- [pypdf](https://pypdf.readthedocs.io/) for PDF text extraction
- [Streamlit](https://streamlit.io/) for the web interface

## Future ideas

- Persistent vector storage with Chroma or FAISS, so documents are not re-embedded on every run
- Page numbers in the metadata, so answers can cite the page as well as the file
- Chat history that carries across questions
- Support for more file types

## Acknowledgements

This project is not my original work. It is based on the work and teaching of **[Mohamad Hachem](https://github.com/Mohamad-Hachem)**, and I am grateful to him for sharing it. Full credit for the original ideas and approach goes to him, and I encourage you to take a look at his other projects, including [RAG_AI_SYSTEM](https://github.com/Mohamad-Hachem/RAG_AI_SYSTEM).

I built this repository by following along and adapting it as part of my own learning.

## Notes

- Each run re-embeds all documents, which makes a small number of OpenAI API calls.
- Text is extracted per PDF without page numbers, so answers cite the source file but not the page.
- The `fontTools` warnings from `pypdf` are harmless; `fonttools` is installed to reduce them.
