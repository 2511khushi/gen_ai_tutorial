from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["USER_AGENT"] = "my-custom-agent/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Load the webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/President_of_the_United_States")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print("Number of chunks: ", len(chunks))

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)
print("FAISS index size:", vector_store.index.ntotal)

# Ask a question
query = "how long does the president's term last?"
docs = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
        print(f"\n{'-' * 80}\nResult {i}:\n{doc.page_content}\n")

# Save and reload the vector store
vector_store.save_local("vector_db")
print("Vector store saved to 'vector_db'")
vector_store = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
print("Vector store reloaded successfully")

