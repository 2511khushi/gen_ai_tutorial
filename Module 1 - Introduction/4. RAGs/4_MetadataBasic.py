import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistence directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db")  # Match the directory from 1_Basics.py

# Define the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Debug: Check the number of documents in the vector store
print(f"Total documents in vector store: {len(db.get()['documents'])}")

# Define the user's question (relevant to The Lord of the Rings)
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Results ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: {doc.page_content}\n")
    print(f"[SOURCE: {doc.metadata['source']}]\n")