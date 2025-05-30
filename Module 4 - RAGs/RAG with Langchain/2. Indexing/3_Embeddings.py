from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Load the web page content
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
documents = loader.load()

# Split the content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print("Number of chunks:", len(chunks))


# Hugging Face Embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chunk_embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
print("HuggingFace embedding dimension:", len(chunk_embeddings[0]))


# OpenAI Embeddings
openai_embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

query_embedding = openai_embeddings_model.embed_query("What is the capital of the United States?")
print("OpenAI query embedding dimension:", len(query_embedding)) 

openai_chunk_embeddings = openai_embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
print("OpenAI chunk embedding dimension:", len(openai_chunk_embeddings[0]))

