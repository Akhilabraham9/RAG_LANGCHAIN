from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from filereader import read_multiple_pdfs, DOC_PATHS

load_dotenv()

# Iniitalize
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Load the documents
docs = read_multiple_pdfs(DOC_PATHS)

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
all_splits = text_splitter.split_documents(docs)
print(f"split documentation into {len(all_splits)} chunks")

# Indexing
vector_store.add_documents(documents = all_splits)
print(f"Indexed {len(all_splits)}")