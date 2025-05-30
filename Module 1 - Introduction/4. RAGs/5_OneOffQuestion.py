import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")  # Match the directory from 1_Basics.py

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the user's question
query = "What does Frodo fear the most?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},  # Added score_threshold
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Results ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: {doc.page_content}\n")
    print(f"[SOURCE: {doc.metadata['source']}]\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
)

# Create a prompt and get a response
messages = [
    SystemMessage(content="You are a helpful assistant. Please answer the user's question using the provided documents. If you cannot answer based on the documents, say 'I'm not sure.'"),
    HumanMessage(content=combined_input),
]

# Define the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Get the response
result = model.invoke(messages)

# Print the result
print("\n--- Generated Response ---")
print("Content only:")
print(result.content)