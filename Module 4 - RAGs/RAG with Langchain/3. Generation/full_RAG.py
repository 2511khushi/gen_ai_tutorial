from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
os.environ["USER_AGENT"] = "LangChain RAG Bot"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Load data
loader = WebBaseLoader("https://www.senate.gov/civics/constitution_item/constitution.htm")
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embedding and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# LLM setup
llm = ChatOpenAI(model_name="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the US Constitution. Use the provided context to answer the question. IMPORTANT: If you are unsure of the answer, say 'I don't know'."),
    ("user", "Question: {question}\nContext: {context}")
])
chain = prompt | llm

# Initial query
query = "Is there any plans to change the US Constitution to allow for term limits on senators?"
docs = retriever.invoke(query)
docs_content = "\n\n".join(doc.page_content for doc in docs)
response = chain.invoke({"question": query, "context": docs_content})
print(response.content)

# Interactive loop
while True:
    query = input("Enter a question: ")
    if query.strip().lower() == "exit":
        break
    docs = retriever.invoke(query)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"question": query, "context": docs_content})
    print(response.content)
