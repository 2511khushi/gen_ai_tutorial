from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_response = llm.invoke("Tell me a joke")
print(llm_response)

# Step 1 : Loading Documents

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

folder_path = "/content/docs"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")


# Step 2 : Splitting Documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)
print(f"Split the documents into {len(splits)} chunks.")


# Step 3 : Creating Embeddings

# Using OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
print(f"Created embeddings for {len(document_embeddings)} document chunks.")

# Using SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
print(document_embeddings[0][:5])  


# Step 4 : Creating the Vector Store
from langchain_chroma import Chroma

collection_name = "my_collection"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted to './chroma_db'")


# Step 5 : Performing Similarity Search

query = "When was GreenGrow Innovations founded?"
search_results = vectorstore.similarity_search(query, k=2)
print(f"\nTop 2 most relevant chunks for the query: '{query}'\n")
for i, result in enumerate(search_results, 1):
    print(f"Result {i}:")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Content: {result.page_content}")
    print()


# Step 6 : Creating a Retriever

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_results = retriever.invoke("When was GreenGrow Innovations founded?")
print(retriever_results)


# Step 7 : Creating the RAG Chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "When was GreenGrow Innovations founded?"
response = rag_chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {response}")



